import asyncio
import contextlib
import logging
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anyio import ClosedResourceError, EndOfStream, create_memory_object_stream
from pydantic_ai import (
    PartEndEvent,
    PartStartEvent,
    TextPart,
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
    async def send(self, event: Any) -> Any:
        if isinstance(event, str):
            await self.tx.send(PartStartEvent(part=TextPart(content=event), index=-1))
            await self.tx.send(PartEndEvent(part=TextPart(content=event), index=-1))
            return None
        return await self._send(event)

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
