import asyncio
import logging
import signal
from typing import override

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
from arox.ui.io import (
    AbstractIOAdapter,
    AdapterIOInterface,
    ChatInputEvent,
    StepDoneEvent,
)
from arox.utils import UserInputGenerator

logger = logging.getLogger(__name__)


class TextIOAdapter(AbstractIOAdapter):
    def setup(self, agent):
        if hasattr(agent, "command_manager"):
            completer = CommandCompleter(agent.command_manager)
            self.user_input = UserInputGenerator(completer=completer)
        else:
            self.user_input = UserInputGenerator()

    @override
    async def start(self):
        if self._started:
            return
        self._started = True

        def sigint_handler(signum, frame):
            logger.info("Received SIGINT, cancelling current step...")
            for adapter_io in self.adapter_ios:
                adapter_io.cancel_task()

        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, sigint_handler)

        try:
            async with asyncio.TaskGroup() as tg:
                self._tg = tg
                for adapter_io in self.adapter_ios:
                    tg.create_task(self._process_io(adapter_io))

                # Keep the task group alive
                while True:
                    await asyncio.sleep(1)
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

    async def _flush_stdin(self):
        import sys

        await asyncio.sleep(0.1)
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, Exception):
            pass

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
                    except (EOFError, KeyboardInterrupt):
                        reply["deferred_tools"][key] = ""
                        await self._flush_stdin()
            if event.exception_input.exception is not None:
                print(
                    f"An error occurred: {event.exception_input.exception}\nDo you want to continue? (y/n)"
                )
                try:
                    line = await self.user_input()
                    reply["exception_input"] = {"retry": line.strip().lower() == "y"}
                except (EOFError, KeyboardInterrupt):
                    reply["exception_input"] = {"retry": False}
                    await self._flush_stdin()
            if event.normal_input.request:
                try:
                    line = await self.user_input()
                    reply["normal_input"] = {"user_input": line}
                except (EOFError, KeyboardInterrupt):
                    reply["normal_input"] = {"user_input": None}
                    await self._flush_stdin()
            event.set_reply(reply)
        else:
            print(f"\nUnexpected event type: {event.__class__.__name__}\n")

    @override
    async def handle_event(self, adapter_io: AdapterIOInterface, event):
        await self._handle_output(event)
