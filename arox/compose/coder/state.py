import logging
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import UserPromptPart

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
        self.project_manager = project.ProjectManager(self.workspace, agent)
        self.chat_files.set_candidate_generator(self.project_manager.get_tracked_files)

    async def _parts_to_update(self) -> list[UserPromptPart]:
        new_parts = await super()._parts_to_update()
        file_list = "\n".join(self.project_manager.get_tracked_files())
        new_parts.append(
            UserFileListPart(content=f"<file_list>\n{file_list}\n</file_list>\n")
        )
        return new_parts
