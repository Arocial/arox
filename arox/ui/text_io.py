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

from arox.core.chat import (
    ChatAgent,
    ChatInputEvent,
    ChatInputReply,
    StepDoneEvent,
)
from arox.core.composer import Composer
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)
from arox.core.plugin import CommandCompleter
from arox.utils import UserInputGenerator

logger = logging.getLogger(__name__)


class TextIOAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.user_input: UserInputGenerator = UserInputGenerator()

    async def register_composer(self, composer: Composer):
        await super().register_composer(composer)

        main_agent = composer.main_agent
        completer = CommandCompleter(
            main_agent.command_manager.router, agent=main_agent
        )
        self.user_input = UserInputGenerator(completer=completer)

    async def __aenter__(self):
        def sigint_handler(signum, frame):
            logger.info("Received SIGINT, cancelling current step...")
            for composer in self.composers.values():
                for agent in composer.all_agents().values():
                    if isinstance(agent, ChatAgent):
                        agent.cancel_foreground_task()

        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, sigint_handler)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint_handler)

    async def _flush_stdin(self):
        import sys

        await asyncio.sleep(0.1)
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, Exception):
            pass

    async def _handle_output(self, adapter_io: IOEndpoint, event):
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
            deferred_answers: dict[str, str | None] = {}
            user_input: str | None = None
            retry = False
            for key, tool in event.deferred_tools.items():
                print(f"\n[Agent asks]: {tool.question}")
                try:
                    deferred_answers[key] = await self.user_input()
                except (EOFError, KeyboardInterrupt):
                    deferred_answers[key] = ""
                    await self._flush_stdin()
            if event.exception_input.exception is not None:
                print(
                    f"An error occurred: {event.exception_input.exception}\nDo you want to continue? (y/n)"
                )
                try:
                    line = await self.user_input()
                    retry = line.strip().lower() == "y"
                except (EOFError, KeyboardInterrupt):
                    retry = False
                    await self._flush_stdin()
            if event.normal_input.request:
                agent = self._find_agent(adapter_io)
                while True:
                    try:
                        line = await self.user_input()
                    except (EOFError, KeyboardInterrupt):
                        user_input = None
                        await self._flush_stdin()
                        break
                    if line.startswith("/") and agent is not None:
                        cmd_reply = await agent.command_manager.try_handle_slash(line)
                        if cmd_reply is not None and cmd_reply.output:
                            print(cmd_reply.output)
                        continue
                    user_input = line
                    break
            await adapter_io.send(
                ChatInputReply(
                    req_id=event.req_id,
                    deferred_answers=deferred_answers,
                    user_input=user_input,
                    retry=retry,
                )
            )
        else:
            print(f"\nUnexpected event type: {event.__class__.__name__}\n")

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event):
        await self._handle_output(adapter_io, event)
