import asyncio
import contextlib
import logging
import math
import signal
from abc import ABC, abstractmethod
from typing import Any, override

from anyio import EndOfStream, create_memory_object_stream
from pydantic_ai import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

from arox.commands import CommandCompleter
from arox.utils import user_input_generator

logger = logging.getLogger(__name__)


class AgentIOInterface(ABC):
    @abstractmethod
    async def agent_send(self, event):
        pass

    @contextlib.asynccontextmanager
    async def chat_round(self):
        yield

    @abstractmethod
    async def agent_receive(self):
        pass

    @abstractmethod
    async def agent_wait_reply(self, event):
        pass

    @abstractmethod
    async def run_cancellable(self, task):
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

    @override
    @contextlib.asynccontextmanager
    async def chat_round(self):
        try:
            yield await self.agent_wait_reply(None)
        finally:
            await self.agent_send(StepDoneEvent())

    @override
    async def run_cancellable(self, task):
        await self.adapter.run_cancellable(task)

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
    async def agent_wait_reply(self, prompt):
        wrap_event = NeedReplyEvent(prompt)
        await self.agent_send(wrap_event)
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
        await self._stack.enter_async_context(self.adapter_event_tx)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()


class StepDoneEvent:
    pass


class NeedReplyEvent:
    def __init__(self, nested_event):
        self.nested_event = nested_event


class AbstractIOAdapter(ABC):
    def __init__(self, adapter_io: AdapterIOInterface):
        self.adapter_io = adapter_io
        adapter_io.set_adapter(self)

    def setup(self, agent):
        pass

    @abstractmethod
    async def start(self):
        pass

    async def run_cancellable(self, task):
        await task


class TextIOAdapter(AbstractIOAdapter):
    def setup(self, agent):
        if hasattr(agent, "command_manager"):
            completer = CommandCompleter(agent.command_manager)

            def user_input():
                return user_input_generator(completer=completer)

            self.user_input = user_input
        else:
            self.user_input = user_input_generator

    async def run_cancellable(self, task):
        step_task = asyncio.create_task(task)
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler(signum, frame):
            logger.info("Received SIGINT, cancelling current step...")
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(step_task.cancel)

        signal.signal(signal.SIGINT, sigint_handler)

        try:
            await step_task
        except asyncio.CancelledError:
            print("\n[Step cancelled by user]")
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

    async def _handle_output(self, event):
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, (TextPart, ThinkingPart)):
                print(f"{part.part_kind}: ", end="")
                print(f"{part.content}", end="")
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, (TextPartDelta, ThinkingPartDelta)):
                print(event.delta.content_delta, end="")
            elif isinstance(event.delta, ToolCallPartDelta):
                print(event.delta.args_delta, end="")
        elif isinstance(event, PartEndEvent):
            print()
        elif isinstance(event, FunctionToolResultEvent):
            print(
                f"tool result: {event.tool_call_id!r} returned => {str(event.result.content)[:100]}\n"
            )
        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            print(
                f"tool call: {part.tool_call_id}: {part.tool_name} args: {str(part.args)[:100]}"
            )
        elif isinstance(event, (FinalResultEvent, StepDoneEvent)):
            pass
        elif isinstance(event, NeedReplyEvent):
            try:
                line = await self.user_input()
                await self.adapter_io.adapter_send(line)
            except EOFError as e:
                await self.adapter_io.adapter_send(e)
        else:
            print(f"\nUnexpected event type: {event.__class__.__name__}\n")

    @override
    async def start(self):
        async with self.adapter_io:
            try:
                while True:
                    event = await self.adapter_io.adapter_receive()
                    await self._handle_output(event)
            except EndOfStream:
                pass
