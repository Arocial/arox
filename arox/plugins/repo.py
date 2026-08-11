import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import git
from pydantic_ai import ModelMessage, ModelRequest, RunContext, UserPromptPart

from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.plugins.slots import PROJECT_FILES

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AddFileListEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("add_file_list",)
    description: ClassVar[str] = "Add tracked files list to context - /add_file_list"


class RepoPlugin(Plugin):
    def __init__(self, runtime: "AgentRuntime"):
        super().__init__(runtime)
        self.workspace = runtime.workspace
        self._pending_project_file_list = False

        # Register as a provider for "project_files"
        self.runtime.provide_slot(PROJECT_FILES, self._get_tracked_files)

    def _get_tracked_files(self):
        try:
            repo = git.Repo(self.workspace, search_parent_directories=True)
            if not repo.working_dir:
                return []

            repo_root = Path(repo.working_dir).resolve()
            workspace_resolved = self.workspace.resolve()

            files = repo.git.ls_files(str(workspace_resolved)).splitlines()

            if repo_root == workspace_resolved:
                return sorted(files)

            normalized_files = []
            for f in files:
                full_path = repo_root / f
                try:
                    rel_path = full_path.relative_to(workspace_resolved)
                    normalized_files.append(str(rel_path))
                except ValueError:
                    continue

            return sorted(normalized_files)
        except (git.InvalidGitRepositoryError, git.GitCommandError) as e:
            logger.debug(f"Failed to get git tracked files: {e}")
            return []

    def add_project_files(self):
        self._pending_project_file_list = True

    def commands(self):
        return [CommandSpec(AddFileListEvent, self.handle_add_file_list)]

    async def handle_add_file_list(self, event: AddFileListEvent):
        self.add_project_files()

    async def history_processor(
        self, ctx: RunContext[Any], messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if messages and isinstance(messages[-1], ModelRequest):
            if self._pending_project_file_list:
                self._pending_project_file_list = False
                file_list = "\n".join(self._get_tracked_files())
                if file_list:
                    extra_content = [
                        f"\nFiles tracked in VC of current project:\n<file_list>{file_list}\n</file_list>\n"
                    ]
                    new_part = UserPromptPart(content=extra_content)
                    last_request = messages[-1]
                    parts = list(last_request.parts)
                    parts.append(new_part)
                    last_request.parts = parts
        return messages
