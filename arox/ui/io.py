import asyncio
import contextlib
import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

from anyio import ClosedResourceError, EndOfStream, create_memory_object_stream
from pydantic_ai import (
    PartEndEvent,
    PartStartEvent,
    TextPart,
)

if TYPE_CHECKING:
    from arox.core.composer import Composer

logger = logging.getLogger(__name__)


class AgentIOInterface(ABC):
    @abstractmethod
    async def agent_send(self, event):
        pass

    @abstractmethod
    async def agent_receive(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AdapterIOInterface(ABC):
    @abstractmethod
    async def adapter_send(self, reply):
        pass

    @abstractmethod
    async def adapter_receive(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AgentIOEndpoint(AgentIOInterface):
    def __init__(self, tx, rx):
        self.tx = tx
        self.rx = rx
        self._stack = contextlib.AsyncExitStack()

    @override
    async def agent_send(self, event):
        if isinstance(event, str):
            await self.tx.send(PartStartEvent(part=TextPart(content=event), index=-1))
            await self.tx.send(PartEndEvent(part=TextPart(content=event), index=-1))
        else:
            await self.tx.send(event)

    @override
    async def agent_receive(self):
        return await self.rx.receive()

    async def __aenter__(self):
        await self._stack.enter_async_context(self.tx)
        await self._stack.enter_async_context(self.rx)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()


class AdapterIOEndpoint(AdapterIOInterface):
    def __init__(self, tx, rx):
        self.tx = tx
        self.rx = rx
        self._stack = contextlib.AsyncExitStack()

    @override
    async def adapter_send(self, reply):
        await self.tx.send(reply)

    @override
    async def adapter_receive(self):
        return await self.rx.receive()

    async def __aenter__(self):
        await self._stack.enter_async_context(self.tx)
        await self._stack.enter_async_context(self.rx)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()


def create_io_channel() -> tuple[AgentIOEndpoint, AdapterIOEndpoint]:
    agent_tx, adapter_rx = create_memory_object_stream[Any](math.inf)
    adapter_tx, agent_rx = create_memory_object_stream[Any](math.inf)
    return AgentIOEndpoint(agent_tx, agent_rx), AdapterIOEndpoint(
        adapter_tx, adapter_rx
    )


class AbstractIOAdapter(ABC):
    def __init__(self):
        self.composers: dict[str, Composer] = {}
        self._composer_tasks: dict[Any, list[asyncio.Task]] = {}
        self._tg: asyncio.TaskGroup = asyncio.TaskGroup()

    async def register_composer(self, composer: "Composer"):
        self.composers[composer.id] = composer

    async def _process_io(self, adapter_io: AdapterIOInterface):
        try:
            while True:
                event = await adapter_io.adapter_receive()
                await self.handle_event(adapter_io, event)
        except (EndOfStream, ClosedResourceError):
            pass

    @abstractmethod
    async def handle_event(self, adapter_io: AdapterIOInterface, event: Any):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
