import logging
from collections.abc import AsyncIterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import (
    AgentStreamEvent,
    ModelMessage,
    RunContext,
)

from arox.agent_patterns.example_parser import parse_example_yaml

if TYPE_CHECKING:
    from arox.agent_patterns.llm_base import AgentDeps

logger = logging.getLogger(__name__)


class SimpleState:
    def __init__(
        self,
        agent,
    ):
        self.agent = agent
        self.system_prompt = self.agent.system_prompt
        self.example_messages = []

        examples_file = getattr(self.agent.agent_config, "examples", None)
        if examples_file:
            examples_path = self.agent.config_parser.find_config(Path(examples_file))
            if examples_path:
                with open(examples_path, "r") as f:
                    self.example_messages = parse_example_yaml(f.read())

        self.workspace = self.agent.workspace
        self.reset()

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    def reset(self):
        self.message_history = self.example_messages

    async def handle_event(
        self, ctx: RunContext["AgentDeps"], events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.agent_io.agent_send(event)
