import logging
import mimetypes
from collections.abc import AsyncIterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import (
    AgentStreamEvent,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    UserPromptPart,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

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
            pending_text_files, pending_binary, pending_project_file_list = (
                self.project_manager.consume_pending()
            )

            extra_content = []
            if pending_project_file_list:
                file_list = "\n".join(self.project_manager._get_tracked_files())
                if file_list:
                    extra_content.append(
                        f"\nFiles tracked in VC of current project:\n{file_list}\n"
                    )

            for path, data in pending_binary.items():
                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "application/octet-stream"
                extra_content.append(BinaryContent(data=data, media_type=mime_type))  # type: ignore

            if extra_content:
                new_part = UserPromptPart(content=extra_content)
                last_request = messages[-1]
                parts = list(last_request.parts)
                parts.append(new_part)
                last_request.parts = parts

            if pending_text_files:
                import uuid

                tool_call_parts = []
                tool_return_parts = []

                for path, content in pending_text_files.items():
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_call_parts.append(
                        ToolCallPart(
                            tool_name="read",
                            args={"path": path},
                            tool_call_id=tool_call_id,
                        )
                    )

                    tool_return_value = {
                        "file_name": path,
                        "content": f"<file path={path}>\n{content}\n</file>\n",
                    }

                    tool_return_parts.append(
                        ToolReturnPart(
                            tool_name="read",
                            content=tool_return_value,
                            tool_call_id=tool_call_id,
                        )
                    )

                if tool_call_parts and tool_return_parts:
                    messages.append(ModelResponse(parts=tool_call_parts))
                    messages.append(ModelRequest(parts=tool_return_parts))
        return messages

    def reset(self):
        self.message_history = self.example_messages

    async def handle_event(
        self, ctx: RunContext["AgentDeps"], events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.agent_io.agent_send(event)
