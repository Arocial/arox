import logging
import sys
from typing import Any, override

from pydantic_ai import AgentRunResultEvent

from arox.core.io import AbstractIOAdapter, IOEndpoint
from arox.core.session import ErrorEvent

logger = logging.getLogger(__name__)


class HeadlessIOAdapter(AbstractIOAdapter):
    """Non-interactive adapter that prints one task's final result.

    The final answer is taken from the ``AgentRunResultEvent`` emitted at the
    end of a step (its ``result.output``); streaming text, tool calls and
    thinking are dropped. The app executes the prompt directly with a
    ``AgentRuntime`` rather than starting an interactive chat loop.
    """

    def __init__(self):
        super().__init__()
        self.error: BaseException | None = None

    @override
    async def handle_event(self, adapter_ep: IOEndpoint, event: Any):
        if isinstance(event, AgentRunResultEvent):
            output = event.result.output
            if output:
                sys.stdout.write(str(output))
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        if isinstance(event, ErrorEvent):
            self.error = RuntimeError(event.error)
            return
