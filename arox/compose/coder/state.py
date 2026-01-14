import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    PartEndEvent,
    RunContext,
    TextPart,
    UserPromptPart,
)

from arox.agent_patterns.state import SimpleState
from arox.codebase import project

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class UserFileListPart(UserPromptPart):
    user_part_kind: Literal["file_list"] = "file_list"


class CoderState(SimpleState):
    def __init__(
        self,
        agent,
    ):
        super().__init__(agent)
        self.diff_agent = self.agent.context.get("diff_agent")
        self.project_manager = project.ProjectManager(agent)

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        if messages and isinstance(messages[-1], ModelRequest):
            pending_files = self.project_manager.consume_pending()
            if pending_files:
                new_part = UserPromptPart(
                    content=pending_files
                )
                last_request = messages[-1]
                parts = list(last_request.parts)
                parts.append(new_part)
                last_request.parts = parts
        return messages

    async def handle_event(
        self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
                await self.project_manager.update_contents(
                    event.part.content, self.diff_agent
                )
            await ctx.deps.io_channel.send(event)
