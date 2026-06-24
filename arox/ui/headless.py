import logging
import sys
from typing import Any, override

from pydantic_ai import AgentRunResultEvent

from arox.core.chat import ChatInputReply, ChatInputRequest
from arox.core.io import AbstractIOAdapter, IOEndpoint

logger = logging.getLogger(__name__)


class HeadlessIOAdapter(AbstractIOAdapter):
    """Non-interactive adapter: feed one prompt, print the final result, exit.

    The final answer is taken from the ``AgentRunResultEvent`` emitted at the
    end of a step (its ``result.output``); streaming text, tool calls and
    thinking are dropped. On the first ChatInputRequest it replies with the
    prompt; on any subsequent request it replies with abort so
    ChatAgent.run() exits cleanly.
    """

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt
        self._consumed = False
        self.error: BaseException | None = None

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event: Any):
        if isinstance(event, AgentRunResultEvent):
            output = event.result.output
            if output:
                sys.stdout.write(str(output))
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        if isinstance(event, ChatInputRequest):
            if event.pending_exception is not None:
                self.error = event.pending_exception
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
