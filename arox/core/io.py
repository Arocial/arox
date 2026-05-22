import asyncio
import contextlib
import logging
import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anyio import ClosedResourceError, EndOfStream, create_memory_object_stream
from pydantic_ai import (
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

if TYPE_CHECKING:
    from arox.core.composer import Composer

logger = logging.getLogger(__name__)


@dataclass
class RequestEvent:
    """Marker base class for events that expect a matching :class:`ReplyEvent`.

    When passed to :meth:`IOEndpoint.send`, the call awaits a reply with
    the same ``req_id`` and returns it. ``RequestEvent`` is direction-
    agnostic.
    """

    req_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ReplyEvent:
    """Reply to a :class:`RequestEvent`. ``req_id`` must match the request."""

    req_id: str


class _BaseIOEndpoint:
    """Shared send/receive plumbing with request/reply correlation.

    Reply correlation relies on someone calling :meth:`_receive`
    concurrently with senders awaiting their requests; in the agent runtime
    this is the adapter event loop and the adapter's ``_process_io`` task
    respectively.
    """

    def __init__(self, tx, rx):
        self.tx = tx
        self.rx = rx
        self._stack = contextlib.AsyncExitStack()
        self._pending: dict[str, asyncio.Future[ReplyEvent]] = {}

    async def _send(self, event: Any) -> Any:
        if isinstance(event, RequestEvent):
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[ReplyEvent] = loop.create_future()
            self._pending[event.req_id] = fut
            try:
                await self.tx.send(event)
                return await fut
            finally:
                self._pending.pop(event.req_id, None)
        await self.tx.send(event)
        return None

    async def _receive(self) -> Any:
        while True:
            event = await self.rx.receive()
            if isinstance(event, ReplyEvent):
                fut = self._pending.pop(event.req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(event)
                else:
                    logger.warning(
                        "Reply for unknown or expired req_id %s (%s)",
                        event.req_id,
                        type(event).__name__,
                    )
                continue
            return event

    async def __aenter__(self):
        await self._stack.enter_async_context(self.tx)
        await self._stack.enter_async_context(self.rx)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        await self._stack.aclose()


class IOEndpoint(_BaseIOEndpoint):
    _SYNTHETIC_INDEX_MIN = -(2**31)

    def __init__(self, tx, rx):
        super().__init__(tx, rx)
        self._synthetic_index = 0

    def _next_synthetic_index(self) -> int:
        self._synthetic_index -= 1
        if self._synthetic_index < self._SYNTHETIC_INDEX_MIN:
            self._synthetic_index = -1
        return self._synthetic_index

    async def send(self, event: Any) -> Any:
        if isinstance(event, str):
            index = self._next_synthetic_index()
            part = TextPart(content=event)
            await self.tx.send(PartStartEvent(part=part, index=index))
            await self.tx.send(PartEndEvent(part=part, index=index))
            return None
        return await self._send(event)

    @contextlib.asynccontextmanager
    async def text_stream(self):
        """Stream a single TextPart as PartStart + many PartDelta + PartEnd
        under one synthetic index. Yields an async ``write(delta)`` callable.
        Use when a logical block of text arrives in pieces (e.g. line-by-line
        shell output) so the UI groups it as one message instead of N."""
        index = self._next_synthetic_index()
        part = TextPart(content="")
        await self.tx.send(PartStartEvent(part=part, index=index))
        try:

            async def write(delta: str) -> None:
                if not delta:
                    return
                await self.tx.send(
                    PartDeltaEvent(
                        delta=TextPartDelta(content_delta=delta), index=index
                    )
                )

            yield write
        finally:
            await self.tx.send(PartEndEvent(part=part, index=index))

    async def receive(self) -> Any:
        return await self._receive()


def create_io_channel() -> tuple[IOEndpoint, IOEndpoint]:
    agent_tx, adapter_rx = create_memory_object_stream[Any](math.inf)
    adapter_tx, agent_rx = create_memory_object_stream[Any](math.inf)
    return IOEndpoint(agent_tx, agent_rx), IOEndpoint(adapter_tx, adapter_rx)


class AbstractIOAdapter(ABC):
    def __init__(self):
        self.composers: dict[str, Composer] = {}
        self._composer_tasks: dict[Any, list[asyncio.Task]] = {}
        self._tg: asyncio.TaskGroup = asyncio.TaskGroup()

    async def register_composer(self, composer: "Composer"):
        self.composers[composer.id] = composer

    def _find_agent(self, adapter_io: IOEndpoint):
        """Locate the agent that owns ``adapter_io`` across registered composers."""
        for composer in self.composers.values():
            for agent in composer.all_agents().values():
                if getattr(agent, "adapter_io", None) is adapter_io:
                    return agent
        return None

    async def _process_io(self, adapter_io: IOEndpoint):
        try:
            while True:
                event = await adapter_io.receive()
                await self.handle_event(adapter_io, event)
        except (EndOfStream, ClosedResourceError):
            pass

    @abstractmethod
    async def handle_event(self, adapter_io: IOEndpoint, event: Any):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class IOHost:
    """Owns one side of an :func:`create_io_channel` pair and a receive loop.

    Subclasses (currently :class:`LLMBaseAgent` and :class:`Composer`)
    add their own domain on top: tool execution, command handling, etc.
    The base just wires the channel, drives ``io_adapter._process_io`` for
    the adapter side, and dispatches inbound :class:`RequestEvent` to
    handlers registered via :meth:`register_request_handler`.
    """

    def __init__(self, io_adapter: "AbstractIOAdapter"):
        self.agent_io, self.adapter_io = create_io_channel()
        self.io_adapter = io_adapter
        self._stack = contextlib.AsyncExitStack()
        self._request_handlers: dict[type[RequestEvent], Callable[[Any], Any]] = {}

    def register_request_handler(
        self,
        event_type: type[RequestEvent],
        handler: Callable[[Any], Any],
    ) -> None:
        """Register a handler for a :class:`RequestEvent` subclass.

        Handlers may be sync or async; the receiver loop awaits coroutines.
        If the handler returns a :class:`ReplyEvent`, it is sent back;
        otherwise a default :class:`ReplyEvent` is sent.
        """
        self._request_handlers[event_type] = handler

    async def _receive_loop(self) -> None:
        while True:
            try:
                event = await self.agent_io.receive()
            except (EndOfStream, ClosedResourceError):
                return
            if isinstance(event, RequestEvent):
                handler = self._request_handlers.get(type(event))
                if handler is None:
                    logger.warning(
                        "No handler registered for RequestEvent %s",
                        type(event).__name__,
                    )
                    await self.agent_io.send(ReplyEvent(req_id=event.req_id))
                    continue
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, ReplyEvent):
                        result.req_id = event.req_id
                        await self.agent_io.send(result)
                    else:
                        await self.agent_io.send(ReplyEvent(req_id=event.req_id))
                except Exception:
                    logger.exception(
                        "Error handling RequestEvent %s", type(event).__name__
                    )
            else:
                logger.debug(
                    "Ignoring non-RequestEvent on adapter->host channel: %r",
                    type(event).__name__,
                )

    async def __aenter__(self):
        self._tg = asyncio.TaskGroup()
        await self._stack.enter_async_context(self._tg)
        self._tg.create_task(self.io_adapter._process_io(self.adapter_io))
        await self._stack.enter_async_context(self.agent_io)
        await self._stack.enter_async_context(self.adapter_io)
        self._tg.create_task(self._receive_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()
