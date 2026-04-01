import asyncio
import logging
import signal
from typing import override

from anyio import EndOfStream
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

from arox.core.plugin import CommandCompleter
from arox.ui.io import AbstractIOAdapter, ChatInputEvent, StepDoneEvent
from arox.utils import user_input_generator

logger = logging.getLogger(__name__)


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
            return await step_task
        except asyncio.CancelledError:
            print("\n[Step cancelled by user]")
            return None
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
                if event.delta.content_delta:
                    print(event.delta.content_delta, end="")
            elif isinstance(event.delta, ToolCallPartDelta):
                if event.delta.args_delta:
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
        elif isinstance(event, ChatInputEvent):
            reply = {}
            if event.deferred_tools:
                reply["deferred_tools"] = {}
                for key, tool in event.deferred_tools.items():
                    print(f"\n[Agent asks]: {tool.question}")
                    try:
                        line = await self.user_input()
                        reply["deferred_tools"][key] = line
                    except EOFError:
                        reply["deferred_tools"][key] = ""
            if event.exception_input.exception is not None:
                print(
                    f"An error occurred: {event.exception_input.exception}\nDo you want to continue? (y/n)"
                )
                try:
                    line = await self.user_input()
                    reply["exception_input"] = {"retry": line.strip().lower() == "y"}
                except EOFError:
                    reply["exception_input"] = {"retry": False}
            if event.normal_input.request:
                try:
                    line = await self.user_input()
                    reply["normal_input"] = {"user_input": line}
                except EOFError:
                    reply["normal_input"] = {"user_input": None}
            event.set_reply(reply)
        else:
            print(f"\nUnexpected event type: {event.__class__.__name__}\n")

    @override
    async def start(self):
        import anyio

        async def process_io(adapter_io):
            async with adapter_io:
                try:
                    while True:
                        event = await adapter_io.adapter_receive()
                        await self._handle_output(event)
                except EndOfStream:
                    pass

        async with anyio.create_task_group() as tg:
            for adapter_io in self.adapter_ios:
                tg.start_soon(process_io, adapter_io)
