import logging
from pathlib import Path
from typing import TYPE_CHECKING

import git

if TYPE_CHECKING:
    from arox.agent_patterns.llm_base import LLMBaseAgent
from arox.codebase.file_edit import FileEdit
from arox.utils import (
    DEFAULT_READ_LIMIT,
    truncate_content,
)

logger = logging.getLogger(__name__)


class ProjectManager:
    def __init__(self, agent: "LLMBaseAgent"):
        self.workspace = agent.workspace
        self.agent = agent
        self._pending_text_files: dict[str, str] = {}
        self._pending_binary_files: dict[str, bytes] = {}
        self.session_files = []
        self.agent.add_local_tool(self.read)
        edit_tool = FileEdit()
        self.agent.add_local_tool(edit_tool.replace_in_file, sequential=True)
        self.agent.add_local_tool(edit_tool.write_to_file, sequential=True)

        self._pending_project_file_list = False

        # Auto read agents.md or agent.md if present (case-insensitive)
        for item in self.workspace.iterdir():
            if item.is_file() and item.name.lower() in ("agents.md", "agent.md"):
                try:
                    self._pending_text_files[item.name] = "".join(
                        self._read_raw(item.name)
                    )
                    self._add_to_session(item.name)
                    break
                except Exception:
                    pass

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

    def candidates(self):
        return self._get_tracked_files()

    def _normalize_path(self, file_path: str) -> Path:
        workspace = self.workspace
        # normalize file path to relative to workspace if it's subtree of workspace, otherwise absolute.
        p = Path(file_path)
        if not p.is_absolute():
            p = (workspace / p).absolute()
        if p.is_relative_to(workspace):
            p = p.relative_to(workspace)
        return p

    def _add_to_session(self, file_path: str):
        if file_path not in self.session_files:
            self.session_files.append(file_path)

    def _read_raw(self, file_path: str) -> list[str]:
        path = self._normalize_path(file_path)
        if self._is_binary_file(path):
            raise Exception(f"Cannot read binary file: {file_path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()

    async def read_by_user(self, file_paths: list[str]):
        for file_path in file_paths:
            try:
                path = self._normalize_path(file_path)
                if self._is_binary_file(path):
                    with open(path, "rb") as f:
                        self._pending_binary_files[file_path] = f.read()
                else:
                    lines = self._read_raw(file_path)
                    self._pending_text_files[file_path] = "".join(lines)
                self._add_to_session(file_path)
            except Exception as e:
                await self.agent.agent_io.agent_send(
                    f"Error reading file {file_path}: {e!s}"
                )

    def consume_pending(self) -> tuple[str, dict[str, bytes]]:
        text_result = ""
        if self._pending_text_files:
            text_result += (
                "User added following files for reference:\n"
                + (
                    "\n".join(
                        [
                            f"<file path={path}>\n{content}\n</file>\n"
                            for path, content in self._pending_text_files.items()
                        ]
                    )
                )
                + "\n"
            )
            self._pending_text_files = {}

        if self._pending_project_file_list:
            file_list = "\n".join(self._get_tracked_files())
            if file_list:
                text_result += (
                    f"\nFiles tracked in VC of current project:\n{file_list}\n"
                )
            self._pending_project_file_list = False

        binary_result = self._pending_binary_files
        self._pending_binary_files = {}

        return text_result, binary_result

    # https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/read.ts
    def read(
        self,
        path: str,
        offset: int = 0,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> dict[str, str]:
        """Reads a file from the local filesystem.
        It's better to read multiple files as a batch that are potentially useful.

        Args:
            path: The path to the file to read.
            offset: The line number to start reading from (0-based).
            limit: The number of lines to read (defaults to 2000).
        """
        result = {"file_name": path}
        try:
            lines = self._read_raw(path)

            truncated = truncate_content(lines, offset, limit)
            content_lines = truncated["lines"]
            last_read_line = truncated["last_read_line"]

            if content_lines:
                result["content"] = (
                    f"<file path={path}>\n{'\n'.join(content_lines)}\n</file>\n"
                )

            if truncated["truncated_by_bytes"]:
                result["truncated"] = (
                    f"Output truncated at {truncated['max_bytes']} bytes. "
                    f"Use 'offset' parameter to read beyond line {last_read_line}"
                )
            elif truncated["has_more_lines"]:
                result["truncated"] = (
                    f"File has more lines. "
                    f"Use 'offset' parameter to read beyond line {last_read_line}"
                )

            self._add_to_session(path)
            return result

        except Exception as e:
            logger.error(f"Error reading file {path}: {e!s}")
            result["error"] = f"Error reading file: {e!s}"
            return result

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is binary using extension and content analysis."""
        ext = path.suffix.lower()
        binary_extensions = {
            ".zip",
            ".tar",
            ".gz",
            ".exe",
            ".dll",
            ".so",
            ".class",
            ".jar",
            ".war",
            ".7z",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".bin",
            ".dat",
            ".obj",
            ".o",
            ".a",
            ".lib",
            ".wasm",
            ".pyc",
            ".pyo",
        }
        if ext in binary_extensions:
            return True

        try:
            file_size = path.stat().st_size
            if file_size == 0:
                return False

            with open(path, "rb") as f:
                # Read first chunk to check for null bytes or high non-printable ratio
                chunk = f.read(4096)
                if not chunk:
                    return False

                if b"\0" in chunk:
                    return True

                non_printable = 0
                for b in chunk:
                    # Non-printable chars: < 9 (TAB), or (between 13 (CR) and 32 (SPACE))
                    if b < 9 or (13 < b < 32):
                        non_printable += 1

                # If >30% non-printable characters, consider it binary
                return non_printable / len(chunk) > 0.3
        except Exception:
            # If we can't read it or stat it, treat it as binary/unreadable
            return True
