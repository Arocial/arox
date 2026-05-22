import logging
import sys
from typing import Any, override

from pydantic_ai import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)

from arox.core.chat import ChatInputEvent, ChatInputReply
from arox.core.io import AbstractIOAdapter, IOEndpoint

logger = logging.getLogger(__name__)


class HeadlessIOAdapter(AbstractIOAdapter):
    """Non-interactive adapter: feed one prompt, stream final answer, exit.

    Only emits the final answer text (gated by FinalResultEvent + TextPart);
    tool calls, thinking, deltas of non-text parts are dropped. On the first
    ChatInputEvent it replies with the prompt; on any subsequent request it
    replies with abort so ChatAgent.run() exits cleanly.
    """

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt
        self._consumed = False
        self._in_final = False
        self.error: BaseException | None = None

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event: Any):
        if isinstance(event, FinalResultEvent):
            self._in_final = True
            return

        if isinstance(event, ChatInputEvent):
            if event.exception_input.exception is not None:
                self.error = event.exception_input.exception
            user_input = None if self._consumed else self.prompt
            self._consumed = True
            await adapter_io.send(
                ChatInputReply(
                    req_id=event.req_id,
                    deferred_answers={k: "" for k in event.deferred_tools},
                    user_input=user_input,
                    retry=False,
                )
            )
            return

        if not self._in_final:
            return

        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart) and event.part.content:
                sys.stdout.write(event.part.content)
                sys.stdout.flush()
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                sys.stdout.write(event.delta.content_delta)
                sys.stdout.flush()
        elif isinstance(event, PartEndEvent):
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._in_final = False
