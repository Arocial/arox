import logging
from collections.abc import AsyncIterable

from pydantic_ai import (
    AgentStreamEvent,
    ModelMessage,
    RunContext,
)

logger = logging.getLogger(__name__)


class SimpleState:
    def __init__(
        self,
        agent,
    ):
        self.agent = agent
        self.system_prompt = self.agent.system_prompt
        self.workspace = self.agent.workspace
        self._result = None
        self.reset()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def message_history(self):
        if self.result:
            return self.result.all_messages()
        else:
            return None

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    def reset(self):
        self.result = None

    async def handle_event(
        self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.io_channel.send(event)
