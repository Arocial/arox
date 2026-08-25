import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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


class IOEndpoint:
    """Channel endpoint for event streaming."""

    _SYNTHETIC_INDEX_MIN = -(2**31)
    _state_lock = threading.RLock()

    def __init__(self):
        self._peer: IOEndpoint | None = None
        self._snapshot_value: Any = None
        self._cached_events: list[Any] = []
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False
        self._disconnected = asyncio.Event()
        self._synthetic_index = 0

    def pair(self, peer: "IOEndpoint") -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("IO endpoint is closed.")
            if peer._closed:
                raise RuntimeError("IO peer is closed.")
            if self._peer is not None:
                raise RuntimeError("IO endpoint is already connected.")
            if peer._peer is not None:
                raise RuntimeError("IO peer is already connected.")
            self._peer = peer
            peer._peer = self
            self._disconnected.clear()
            peer._disconnected.clear()
            self._replay_to(peer)
            peer._replay_to(self)

    @property
    def peer(self) -> "IOEndpoint | None":
        return self._peer

    def disconnect(self) -> None:
        with self._state_lock:
            peer = self._peer
            if peer is None:
                return
            self._peer = None
            self._disconnected.set()
            if peer._peer is self:
                peer._peer = None
                peer._disconnected.set()

    async def wait_disconnected(self) -> None:
        """Wait until this endpoint is no longer paired with its peer."""
        await self._disconnected.wait()

    def _replay_to(self, peer: "IOEndpoint") -> None:
        if self._snapshot_value is not None:
            peer._inbox.put_nowait(SnapshotEvent(self._snapshot_value))
        for event in self._cached_events:
            peer._inbox.put_nowait(event)

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

    async def send(self, event: Any) -> None:
        if isinstance(event, str):
            with self._state_lock:
                index = self._next_synthetic_index()
                part = TextPart(content=event)
                self._send_event(PartStartEvent(part=part, index=index))
                self._send_event(PartEndEvent(part=part, index=index))
            return None
        self._send_event(event)
        return None

    async def receive(self) -> Any:
        while True:
            if self._closed:
                raise StopAsyncIteration
            event = await self._inbox.get()
            if event is _STREAM_CLOSED:
                raise StopAsyncIteration
            return event

    def snapshot(self, snapshot: Any) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("IO endpoint is closed.")
            self._snapshot_value = snapshot
            self._cached_events.clear()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self.disconnect()
            while not self._inbox.empty():
                self._inbox.get_nowait()
            self._inbox.put_nowait(_STREAM_CLOSED)
            self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AbstractIOAdapter(ABC):
    def __init__(self):
        self.adapter_ep_to_runtime: dict[IOEndpoint, AgentRuntime] = {}
        self._event_consumer_tasks: dict[IOEndpoint, asyncio.Task[None]] = {}

    async def connect(self, runtime: "AgentRuntime") -> IOEndpoint:
        await self.disconnect(runtime)
        adapter_ep = IOEndpoint()
        agent_ep = runtime.agent_ep
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
        adapter_ep.disconnect()
        self.adapter_ep_to_runtime.pop(adapter_ep, None)
        task = self._event_consumer_tasks.pop(adapter_ep, None)
        adapter_ep.close()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

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
    """Agent-side endpoint with an inbound event dispatch loop.

    The adapter owns the peer endpoint and its consumer. Inbound events are
    dispatched to handlers registered for their concrete event type.
    """

    def __init__(self):
        super().__init__()
        self._event_handlers: dict[type[Any], Callable[[Any], Any]] = {}

    def register_event_handler(
        self,
        event_type: type[Any],
        handler: Callable[[Any], Any],
    ) -> None:
        """Register the single handler for an inbound event type.

        Handlers may be synchronous or asynchronous.
        """
        if event_type in self._event_handlers:
            raise RuntimeError(f"Handler already registered for {event_type.__name__}.")
        self._event_handlers[event_type] = handler

    async def _agent_event_loop(self) -> None:
        while True:
            try:
                event = await self.receive()
            except (StopAsyncIteration, RuntimeError):
                return
            handler = self._event_handlers.get(type(event))
            if handler is None:
                logger.warning(
                    "No handler registered for event %s", type(event).__name__
                )
                continue
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Error handling inbound event %s", type(event).__name__
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
