import asyncio
import logging
from pathlib import Path
from typing import Any, override

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from pydantic_ai import (
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
    ChatInputReply,
    ChatInputRequest,
)
from arox.core.completion import CompletionRouter, parse_request
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)
from arox.core.session import ErrorEvent

logger = logging.getLogger(__name__)


class UserInputGenerator:
    def __init__(self, completer=None, input=None, output=None):
        self.input = input
        self.output = output

        arox_dir = Path(".arox")
        arox_dir.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(arox_dir / "history")

        self.kb = KeyBindings()

        @self.kb.add("enter")
        def _(event):  # Enter to submit
            event.current_buffer.validate_and_handle()

        @self.kb.add("escape", "enter")  # Alt+Enter newline
        @self.kb.add("escape", "O", "M")  # Shift+Enter (at least in my konsole)
        def _(event):
            event.current_buffer.insert_text("\n")

        self.session = PromptSession(
            prompt_continuation="> ",
            multiline=True,
            key_bindings=self.kb,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            mouse_support=False,
            completer=completer,
            input=input,
            output=output,
        )

    async def __call__(self):
        return await self.session.prompt_async("\nUser (Ctrl+D to quit): ")


class CommandCompleter(Completer):
    """prompt-toolkit ``Completer`` adapter on top of :class:`CompletionRouter`.

    Owns no completion logic itself — it builds a :class:`CompletionRequest`
    from the buffer's ``Document`` and translates the completion_router's
    :class:`CompletionItem` results back into prompt-toolkit ``Completion``
    objects, computing ``start_position`` from each item's
    ``replace_range``.
    """

    def __init__(
        self, completion_router: "CompletionRouter", *, runtime: Any | None = None
    ):
        self.completion_router = completion_router
        self.runtime = runtime

    def get_completions(self, document, complete_event):
        # Async-only completer; prompt-toolkit drives ``get_completions_async``
        # during async prompt sessions. The sync path yields nothing.
        return iter(())

    async def get_completions_async(self, document, complete_event):
        text = document.text
        if not text or text[0] not in ("/", "@"):
            return
        req = parse_request(text, cursor=document.cursor_position, runtime=self.runtime)
        for item in await self.completion_router.complete(req):
            start, _end = item.replace_range or req.current_token_range
            # start_position is relative to the cursor; document.cursor_position
            # is the absolute cursor in `text`.
            start_position = start - document.cursor_position
            yield Completion(
                item.value,
                start_position=start_position,
                display=item.label or item.value,
                display_meta=item.description or "",
            )


class TextIOAdapter(AbstractIOAdapter):
    def __init__(self, *, input=None, output=None):
        super().__init__()
        self.input = input
        self.output = output
        self.user_inputs: dict[IOEndpoint, UserInputGenerator] = {}

    def _user_input_for(
        self, adapter_ep: IOEndpoint, runtime: Any | None
    ) -> UserInputGenerator:
        user_input = self.user_inputs.get(adapter_ep)
        if user_input is not None:
            return user_input

        completer = None
        if runtime is not None:
            completer = CommandCompleter(
                runtime.command_manager.completion_router,
                runtime=runtime,
            )
        user_input = UserInputGenerator(
            completer=completer,
            input=self.input,
            output=self.output,
        )
        self.user_inputs[adapter_ep] = user_input
        return user_input

    async def on_endpoint_closed(self, adapter_ep: IOEndpoint) -> None:
        self.user_inputs.pop(adapter_ep, None)

    async def _flush_stdin(self):
        import sys

        await asyncio.sleep(0.1)
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, Exception):
            pass

    async def _handle_output(self, adapter_ep: IOEndpoint, event):
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, (TextPart, ThinkingPart)):
                prefix = part.part_kind
                print(f"{prefix}: ", end="")
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
                f"tool result: {event.tool_call_id!r} returned => {str(event.part.content)[:100]}\n"
            )
        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            print(
                f"tool call: {part.tool_call_id}: {part.tool_name} args: {str(part.args)[:100]}"
            )
        elif isinstance(event, ChatInputRequest):
            user_input: str | None = None
            if event.request_normal_input:
                input_generator = self._user_input_for(
                    adapter_ep, self.agent_io_for(adapter_ep)
                )
                try:
                    user_input = await input_generator()
                except (EOFError, KeyboardInterrupt):
                    user_input = None
                    await self._flush_stdin()
            await adapter_ep.send(
                ChatInputReply(
                    req_id=event.req_id,
                    input_content=user_input,
                )
            )
        elif isinstance(event, ErrorEvent):
            print(f"⚠️ An error occurred: {event.error}")
        else:
            logger.debug(f"\nUnknown event type: {event.__class__.__name__}\n")

    @override
    async def handle_event(self, adapter_ep: IOEndpoint, event):
        await self._handle_output(adapter_ep, event)
