import contextlib
import logging
import math
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
    ToolCallPart,
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


class AdapterIOInterface(ABC):
    @abstractmethod
    async def adapter_send(self, reply):
        pass

    @abstractmethod
    async def adapter_receive(self):
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

    @override
    @contextlib.asynccontextmanager
    async def chat_round(self):
        try:
            yield await self.agent_wait_reply(None)
        finally:
            await self.agent_send(StepDoneEvent())

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
    def setup(self, agent):
        pass

    @abstractmethod
    async def start(self):
        pass


class TextIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface):
        self.adapter_io = adapter_io

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


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface):
        self.stream_in, self.stream_out = create_memory_object_stream[str](100)
        self.tool_ids = {}
        self.adapter_io = adapter_io

    @override
    async def start(self):
        async with self.stream_in:
            try:
                while True:
                    event = await self.adapter_io.adapter_receive()
                    if isinstance(event, NeedReplyEvent):
                        pass
                    elif isinstance(event, StepDoneEvent):
                        await self.stream_in.send("data: [DONE]\n\n")
                    else:
                        formatted_events = self._format_event(event)
                        for fmt in formatted_events:
                            await self.stream_in.send(fmt)
            except EndOfStream:
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

            elif isinstance(event.delta, ToolCallPartDelta):
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
        await self.adapter_io.adapter_send(text)
