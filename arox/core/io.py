import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    PartEndEvent,
    PartStartEvent,
    TextPart,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime


_STREAM_CLOSED = object()


@dataclass(frozen=True)
class SnapshotEvent:
    snapshot: Any


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


class IOEndpoint:
    """Channel endpoint with request/reply correlation and event streaming.

    Reply correlation relies on someone calling :meth:`receive` concurrently
    with senders awaiting their requests; in the agent runtime these are the
    event loops owned by :class:`AgentIOEndpoint` and :class:`AbstractIOAdapter`.
    """

    _SYNTHETIC_INDEX_MIN = -(2**31)
    _state_lock = threading.RLock()

    def __init__(self):
        self._peer: IOEndpoint | None = None
        self._snapshot_value: Any = None
        self._cached_events: list[Any] = []
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False
        self._pending: dict[str, asyncio.Future[ReplyEvent]] = {}
        self._synthetic_index = 0

    def pair(self, peer: "IOEndpoint") -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("IO endpoint is closed.")
            if peer._closed:
                raise RuntimeError("IO peer is closed.")
            self._unpair()
            peer._unpair()
            self._peer = peer
            peer._peer = self
            self._replay_to(peer)
            peer._replay_to(self)

    @property
    def peer(self) -> "IOEndpoint | None":
        return self._peer

    def _unpair(self) -> None:
        peer = self._peer
        if peer is None:
            return
        self._peer = None
        if peer._peer is self:
            peer._peer = None
            peer._end_stream()

    def _replay_to(self, peer: "IOEndpoint") -> None:
        if self._snapshot_value is not None:
            peer._inbox.put_nowait(SnapshotEvent(self._snapshot_value))
        for event in self._cached_events:
            peer._inbox.put_nowait(event)

    def _end_stream(self) -> None:
        while not self._inbox.empty():
            self._inbox.get_nowait()
        self._inbox.put_nowait(_STREAM_CLOSED)

    def _send_event(self, event: Any) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("IO endpoint is closed.")
            self._cached_events.append(event)
            if self._peer is not None:
                self._peer._inbox.put_nowait(event)

    def _next_synthetic_index(self) -> int:
        self._synthetic_index -= 1
        if self._synthetic_index < self._SYNTHETIC_INDEX_MIN:
            self._synthetic_index = -1
        return self._synthetic_index

    async def send(self, event: Any) -> Any:
        if isinstance(event, str):
            with self._state_lock:
                index = self._next_synthetic_index()
                part = TextPart(content=event)
                self._send_event(PartStartEvent(part=part, index=index))
                self._send_event(PartEndEvent(part=part, index=index))
            return None
        if isinstance(event, RequestEvent):
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[ReplyEvent] = loop.create_future()
            self._pending[event.req_id] = fut
            try:
                self._send_event(event)
                return await fut
            finally:
                self._pending.pop(event.req_id, None)
        self._send_event(event)
        return None

    async def receive(self) -> Any:
        while True:
            event = await self._inbox.get()
            if event is _STREAM_CLOSED:
                raise StopAsyncIteration
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

    def snapshot(self, snapshot: Any) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("IO endpoint is closed.")
            self._snapshot_value = snapshot
            self._cached_events.clear()
            if self._peer is not None:
                self._peer._inbox.put_nowait(SnapshotEvent(snapshot))

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._unpair()
            self._end_stream()
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AbstractIOAdapter(ABC):
    def __init__(self):
        self.adapter_ep_to_runtime: dict[IOEndpoint, AgentRuntime] = {}
        self._event_consumer_tasks: dict[IOEndpoint, asyncio.Task[None]] = {}

    async def connect(self, runtime: "AgentRuntime") -> IOEndpoint:
        agent_ep = runtime.agent_ep
        current = agent_ep.peer
        if current in self.adapter_ep_to_runtime:
            await self.disconnect(runtime)
        adapter_ep = IOEndpoint()
        agent_ep.pair(adapter_ep)
        self.adapter_ep_to_runtime[adapter_ep] = runtime
        self._event_consumer_tasks[adapter_ep] = asyncio.create_task(
            self._consume_events(adapter_ep)
        )
        return adapter_ep

    async def disconnect(self, runtime: "AgentRuntime") -> None:
        adapter_ep = runtime.agent_ep.peer
        if adapter_ep not in self.adapter_ep_to_runtime:
            return
        adapter_ep.close()
        task = self._event_consumer_tasks.pop(adapter_ep, None)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        self.adapter_ep_to_runtime.pop(adapter_ep, None)

    async def on_runtime_start(self, runtime: "AgentRuntime") -> None:
        await self.connect(runtime)

    async def on_runtime_stop(self, runtime: "AgentRuntime") -> None:
        await self.disconnect(runtime)

    def agent_io_for(self, adapter_ep: IOEndpoint) -> "AgentRuntime":
        return self.adapter_ep_to_runtime[adapter_ep]

    async def _consume_events(self, adapter_ep: IOEndpoint) -> None:
        try:
            while True:
                event = await adapter_ep.receive()
                await self.handle_event(adapter_ep, event)
        except (StopAsyncIteration, RuntimeError):
            return
        finally:
            adapter_ep.close()
            await self.on_endpoint_closed(adapter_ep)
            self.adapter_ep_to_runtime.pop(adapter_ep, None)
            self._event_consumer_tasks.pop(adapter_ep, None)

    @abstractmethod
    async def handle_event(self, adapter_ep: IOEndpoint, event: Any):
        pass

    async def on_endpoint_closed(self, adapter_ep: IOEndpoint) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for r in list(self.adapter_ep_to_runtime.values()):
            await self.disconnect(r)


class AgentIOEndpoint(IOEndpoint):
    """Agent-side endpoint with a request loop.

    The adapter owns the peer endpoint and its consumer. This endpoint dispatches
    inbound :class:`RequestEvent` instances to handlers registered via
    :meth:`register_request_handler`.
    """

    def __init__(self):
        super().__init__()
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

    async def _agent_event_loop(self) -> None:
        while True:
            try:
                event = await self.receive()
            except (StopAsyncIteration, RuntimeError):
                return
            if isinstance(event, RequestEvent):
                handler = self._request_handlers.get(type(event))
                if handler is None:
                    logger.warning(
                        "No handler registered for RequestEvent %s",
                        type(event).__name__,
                    )
                    await self.send(ReplyEvent(req_id=event.req_id))
                    continue
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, ReplyEvent):
                        result.req_id = event.req_id
                        await self.send(result)
                    else:
                        await self.send(ReplyEvent(req_id=event.req_id))
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
        await self._tg.__aenter__()
        self._tg.create_task(self._agent_event_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        finally:
            await self._tg.__aexit__(exc_type, exc_val, exc_tb)
