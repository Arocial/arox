import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Any

from anyio import create_memory_object_stream
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


class AbstractIOAdapter(ABC):
    def setup(self, agent):
        pass

    @abstractmethod
    async def handle_output_stream(self, output_stream_out):
        pass


class IOChannel:
    def __init__(self, adapter: AbstractIOAdapter):
        self.output_stream_in, self.output_stream_out = create_memory_object_stream[
            Any
        ]()
        self.adapter = adapter
        self.async_stack = contextlib.AsyncExitStack()

    async def send(self, event):
        if isinstance(event, str):
            await self.output_stream_in.send(
                PartStartEvent(part=TextPart(content=event), index=-1)
            )
            await self.output_stream_in.send(
                PartEndEvent(part=TextPart(content=event), index=-1)
            )
        else:
            await self.output_stream_in.send(event)

    async def wait_reply(self, event):
        wrap_event = EventNeedReply(event)
        await self.send(wrap_event)
        return await wrap_event.wait()

    async def start(self):
        await self.async_stack.enter_async_context(self.output_stream_in)
        asyncio.create_task(self.adapter.handle_output_stream(self.output_stream_out))

    async def end(self):
        await self.async_stack.aclose()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.end()


class TextIOAdapter(AbstractIOAdapter):
    def setup(self, agent):
        if hasattr(agent, "command_manager"):
            completer = CommandCompleter(agent.command_manager)

            def user_input():
                return user_input_generator(completer=completer)

            self.user_input = user_input
        else:
            self.user_input = user_input_generator

    async def _handle_output(self, event):
        if isinstance(event, PartStartEvent):
            part = event.part
            print(f"{part.part_kind}: ", end="")
            if isinstance(part, (TextPart, ThinkingPart)):
                print(f"{part.content}", end="")
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, (TextPartDelta, ThinkingPartDelta)):
                print(event.delta.content_delta, end="")
            elif isinstance(event.delta, ToolCallPartDelta):
                print(event.delta.args_delta, end="")
        elif isinstance(event, PartEndEvent):
            print("")
        elif isinstance(event, FunctionToolResultEvent):
            print(
                f"tool result: {event.tool_call_id!r} returned => {event.result.content}\n"
            )
        elif isinstance(event, (FunctionToolCallEvent, FinalResultEvent)):
            pass
        elif isinstance(event, EventNeedReply):
            try:
                line = await self.user_input()
                event.set_reply(line)
            except EOFError as e:
                event.set_reply(e)
        else:
            print(f"\nUnexpected event type: {event.__class__.__name__}\n")

    async def handle_output_stream(self, output_stream_out):
        async with output_stream_out:
            async for event in output_stream_out:
                await self._handle_output(event)


class EventNeedReply:
    def __init__(self, nested_event):
        self.nested_event = nested_event
        loop = asyncio.get_running_loop()
        reply_fut = loop.create_future()
        self.future = reply_fut

    def set_reply(self, reply):
        self.future.set_result(reply)

    async def wait(self):
        return await self.future
