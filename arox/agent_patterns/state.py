import logging
import mimetypes
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic_ai import (
    AgentStreamEvent,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    RunContext,
    UserPromptPart,
)

from arox.agent_patterns.example_parser import parse_example_yaml
from arox.codebase import project

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
        self.project_manager = project.ProjectManager(agent)
        self.reset()

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        if messages and isinstance(messages[-1], ModelRequest):
            pending_text, pending_binary = self.project_manager.consume_pending()
            content = []
            if pending_text:
                content.append(pending_text)
            for path, data in pending_binary.items():
                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "application/octet-stream"
                content.append(BinaryContent(data=data, media_type=mime_type))  # type: ignore

            if content:
                new_part = UserPromptPart(content=content)
                last_request = messages[-1]
                parts = list(last_request.parts)
                parts.append(new_part)
                last_request.parts = parts
        return messages

    def reset(self):
        self.message_history = self.example_messages

    async def handle_event(
        self, ctx: RunContext["AgentDeps"], events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.agent_io.agent_send(event)
