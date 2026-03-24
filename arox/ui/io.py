import asyncio
import contextlib
import logging
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, override

from anyio import create_memory_object_stream
from pydantic_ai import (
    PartEndEvent,
    PartStartEvent,
    TextPart,
)

logger = logging.getLogger(__name__)


class AgentIOInterface(ABC):
    @abstractmethod
    async def agent_send(self, event):
        pass

    @abstractmethod
    async def add_tool_input_request(self, question, key):
        pass

    @abstractmethod
    async def get_tool_input_result(self, key) -> str | None:
        pass

    @abstractmethod
    def create_chat_input_event(self) -> "ChatInputEvent":
        pass

    @contextlib.asynccontextmanager
    async def chat_round(self):
        yield

    @abstractmethod
    async def agent_receive(self):
        pass

    @abstractmethod
    async def run_cancellable(self, task):
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

    def set_adapter(self, adapter):
        self.adapter = adapter

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class IOChannel(AgentIOInterface, AdapterIOInterface):
    def __init__(self):
        # agent_event: Agent -> Adapter
        self.agent_event_tx, self.agent_event_rx = create_memory_object_stream[Any](
            math.inf
        )
        # adapter_event: Adapter -> Agent
        self.adapter_event_tx, self.adapter_event_rx = create_memory_object_stream[Any](
            math.inf
        )
        self._stack = contextlib.AsyncExitStack()

        self.chat_input_event = None

    @override
    def create_chat_input_event(self):
        self.chat_input_event = ChatInputEvent()
        return self.chat_input_event

    @override
    @contextlib.asynccontextmanager
    async def chat_round(self):
        assert self.chat_input_event is not None
        await self.chat_input_event.wait()
        try:
            yield
        finally:
            await self.agent_send(self.chat_input_event)
            await self.agent_send(StepDoneEvent())

    @override
    async def add_tool_input_request(self, question, key):
        assert self.chat_input_event is not None
        self.chat_input_event.add_deferred_tool(question, key)

    @override
    async def get_tool_input_result(self, key):
        assert self.chat_input_event is not None
        await self.chat_input_event.wait()
        return self.chat_input_event.get_deferred_tool_input(key)

    @override
    async def run_cancellable(self, task):
        return await self.adapter.run_cancellable(task)

    @override
    async def agent_send(self, event):
        if isinstance(event, str):
            await self.agent_event_tx.send(
                PartStartEvent(part=TextPart(content=event), index=-1)
            )
            await self.agent_event_tx.send(
                PartEndEvent(part=TextPart(content=event), index=-1)
            )
        else:
            await self.agent_event_tx.send(event)

    @override
    async def agent_receive(self):
        return await self.adapter_event_rx.receive()

    @override
    async def adapter_send(self, reply):
        await self.adapter_event_tx.send(reply)

    @override
    async def adapter_receive(self):
        return await self.agent_event_rx.receive()

    async def __aenter__(self):
        await self._stack.enter_async_context(self.agent_event_tx)
        await self._stack.enter_async_context(self.agent_event_rx)
        await self._stack.enter_async_context(self.adapter_event_tx)
        await self._stack.enter_async_context(self.adapter_event_rx)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()


class StepDoneEvent:
    pass


class ChatInputEvent:
    @dataclass
    class DeferredToolInput:
        question: str
        answer: str | None = None

    @dataclass
    class NormalInput:
        request: bool
        user_input: str | None

    @dataclass
    class ExceptionInput:
        exception: BaseException | None = None
        to_continue: bool = False

    def __init__(self):
        self.deferred_tools = OrderedDict[str, self.DeferredToolInput]()
        self.normal_input = self.NormalInput(False, "")
        self.exception_input = self.ExceptionInput()

        loop = asyncio.get_running_loop()
        self.future = loop.create_future()

    def add_deferred_tool(self, question: str, key: str):
        self.deferred_tools[key] = self.DeferredToolInput(question)

    def get_deferred_tool_input(self, key):
        return self.deferred_tools[key].answer

    async def wait(self):
        await self.future

    def generate_request(self):
        return {
            "deferred_tools": {k: t.question for k, t in self.deferred_tools.items()},
            "normal_input": {"request": self.normal_input.request},
            "exception_input": {
                "exception": f"Exception: {self.exception_input.exception}"
                if self.exception_input.exception
                else None
            },
        }

    def set_reply(self, reply: dict):
        if "deferred_tools" in reply:
            for k, v in reply["deferred_tools"].items():
                if k in self.deferred_tools:
                    self.deferred_tools[k].answer = v
        if "exception_input" in reply:
            self.exception_input.to_continue = reply["exception_input"]["to_continue"]
        if "normal_input" in reply:
            self.normal_input.user_input = reply["normal_input"]["user_input"]

        self.future.set_result(True)


class AbstractIOAdapter(ABC):
    def __init__(self, adapter_io: AdapterIOInterface | None = None):
        self.adapter_ios: list[AdapterIOInterface] = []
        if adapter_io:
            self.add_adapter_io(adapter_io)

    def add_adapter_io(self, adapter_io: AdapterIOInterface):
        self.adapter_ios.append(adapter_io)
        adapter_io.set_adapter(self)

    def setup(self, agent):
        pass

    @abstractmethod
    async def start(self):
        pass

    async def run_cancellable(self, task):
        return await task
