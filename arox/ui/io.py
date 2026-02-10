import asyncio
import contextlib
import logging
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
    ToolCallPart,
    ToolCallPartDelta,
)

from arox.commands import CommandCompleter
from arox.utils import user_input_generator

logger = logging.getLogger(__name__)


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
            if isinstance(part, (TextPart, ThinkingPart)):
                print(f"{part.part_kind}: ", end="")
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
                f"tool result: {event.tool_call_id!r} returned => {str(event.result.content)[:100]}\n"
            )
        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            print(
                f"tool call: {part.tool_call_id}: {part.tool_name} args: {str(part.args)[:100]}"
            )
        elif isinstance(event, (FinalResultEvent, StepDoneEvent)):
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


class StepDoneEvent:
    pass


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


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self):
        self.stream_in, self.stream_out = create_memory_object_stream[str](100)
        self.pending_reply_event = None
        self.tool_ids = {}

    async def handle_output_stream(self, output_stream_out):
        async with output_stream_out:
            async with self.stream_in:
                async for event in output_stream_out:
                    if isinstance(event, EventNeedReply):
                        self.pending_reply_event = event
                        try:
                            await event.wait()
                        except Exception:
                            pass
                        self.pending_reply_event = None
                    elif isinstance(event, StepDoneEvent):
                        await self.stream_in.send("data: [DONE]\n\n")
                    else:
                        formatted_events = self._format_event(event)
                        for fmt in formatted_events:
                            await self.stream_in.send(fmt)

                await self.stream_in.send("data: [DONE]\n\n")

    def _format_event(self, event) -> list[str]:
        import json

        events = []

        if isinstance(event, PartStartEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                events.append(
                    f"data: {json.dumps({'type': 'text-start', 'id': f'text_{index}'})}\n\n"
                )
                if part.content:
                    events.append(
                        f"data: {json.dumps({'type': 'text-delta', 'id': f'text_{index}', 'delta': part.content})}\n\n"
                    )

            elif isinstance(part, ThinkingPart):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-start', 'id': f'reasoning_{index}'})}\n\n"
                )
                if part.content:
                    events.append(
                        f"data: {json.dumps({'type': 'reasoning-delta', 'id': f'reasoning_{index}', 'delta': part.content})}\n\n"
                    )

            elif isinstance(part, ToolCallPart):
                self.tool_ids[index] = part.tool_call_id
                events.append(
                    f"data: {json.dumps({'type': 'tool-input-start', 'toolCallId': part.tool_call_id, 'toolName': part.tool_name})}\n\n"
                )
                if part.args and isinstance(part.args, str):
                    events.append(
                        f"data: {json.dumps({'type': 'tool-input-delta', 'toolCallId': part.tool_call_id, 'inputTextDelta': part.args})}\n\n"
                    )

        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            index = event.index

            if isinstance(delta, TextPartDelta):
                events.append(
                    f"data: {json.dumps({'type': 'text-delta', 'id': f'text_{index}', 'delta': delta.content_delta})}\n\n"
                )

            elif isinstance(delta, ThinkingPartDelta):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-delta', 'id': f'reasoning_{index}', 'delta': delta.content_delta})}\n\n"
                )

            elif isinstance(delta, ToolCallPartDelta):
                tool_id = self.tool_ids.get(index)
                if tool_id:
                    events.append(
                        f"data: {json.dumps({'type': 'tool-input-delta', 'toolCallId': tool_id, 'inputTextDelta': delta.args_delta})}\n\n"
                    )

        elif isinstance(event, PartEndEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                events.append(
                    f"data: {json.dumps({'type': 'text-end', 'id': f'text_{index}'})}\n\n"
                )
            elif isinstance(part, ThinkingPart):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-end', 'id': f'reasoning_{index}'})}\n\n"
                )

        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            events.append(
                f"data: {json.dumps({'type': 'tool-input-available', 'toolCallId': part.tool_call_id, 'toolName': part.tool_name, 'input': part.args})}\n\n"
            )

        elif isinstance(event, FunctionToolResultEvent):
            events.append(
                f"data: {json.dumps({'type': 'tool-output-available', 'toolCallId': event.tool_call_id, 'output': event.result.content})}\n\n"
            )

        elif isinstance(event, FinalResultEvent):
            events.append(f"data: {json.dumps({'type': 'finish'})}\n\n")

        return events

    async def output_generator(self):
        async for chunk in self.stream_out:
            yield chunk

    async def submit_user_input(self, text: str):
        for _ in range(10):
            if self.pending_reply_event:
                self.pending_reply_event.set_reply(text)
                return True
            await asyncio.sleep(1)
        logger.warning(f"No input required, ignoring: {text}")
        return False
