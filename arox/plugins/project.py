import asyncio
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import git
from prompt_toolkit.completion import Completion
from rapidfuzz import fuzz

from arox.agent_patterns.plugin import Plugin, ToolDef, command, tool
from arox.utils import DEFAULT_READ_LIMIT, truncate_content

if TYPE_CHECKING:
    from arox.agent_patterns.llm_base import LLMBaseAgent

logger = logging.getLogger(__name__)


class ProjectManager:
    def __init__(self, agent: "LLMBaseAgent"):
        self.workspace = agent.workspace
        self.agent = agent
        self._pending_text_files: dict[str, str] = {}
        self._pending_binary_files: dict[str, bytes] = {}
        self.session_files = []

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

    def consume_pending(self) -> tuple[dict[str, str], dict[str, bytes], bool]:
        text_files = self._pending_text_files
        self._pending_text_files = {}

        project_file_list = self._pending_project_file_list
        self._pending_project_file_list = False

        binary_files = self._pending_binary_files
        self._pending_binary_files = {}

        return text_files, binary_files, project_file_list

    # https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/read.ts
    @tool()
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


_alnum_regex = re.compile(r"(?ui)\W")


class FileEdit:
    @tool(sequential=True)
    async def write_to_file(self, path: str, content: str) -> str:
        """Create or overwrite a file.

        Args:
            path: The path of the file to write to.
            content: The full content to write to the file.

        Returns:
            str: Success message or error description
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing to file: {e!s}"

    @tool(sequential=True)
    async def replace_in_file(self, path: str, old_str: str, new_str: str) -> str:
        """Searches for `old_str` in the file and replaces it with `new_str`.

        Args:
            path: The path of the file to modify.
            old_str: The block of code to be replaced.
                - It must be unique enough to identify the correct section.
            new_str: The full replacement text.
                - This will completely replace the content matched by `old_str`.

        Returns:
            str: A success message if the replacement was successful, or an error message
                 if the file was not found or `old_str` could not be matched.
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return f"File not found: {file_path}"

            orig_content = file_path.read_text()
            content = orig_content

            # Check if search_part contains ...omit lines...
            m, start_pos, end_pos = self._find_with_placeholder(content, old_str)
            if m:
                content = content[:start_pos] + new_str + content[end_pos:]
            else:
                if old_str in content:
                    content = content.replace(old_str, new_str, 1)
                else:
                    content = self._fuzzy_replace(old_str, new_str, content)

            if content:
                file_path.write_text(content)
                msg = f"Successfully updated {file_path}"
            else:
                msg = f"Cannot find a match for passed old_str in {file_path}"
            logger.info(msg)
            return msg
        except Exception as e:
            msg = f"Error replacing in file `{path}` with exception: {e!s}"
            logger.info(msg)
            return msg

    def _match_placeholder(self, content):
        return re.search(
            r"^[^a-zA-Z]*" + re.escape("...omit lines...") + r"[^a-zA-Z]*$",
            content,
            re.MULTILINE,
        )

    def _find_with_placeholder(self, content: str, search_pattern: str) -> tuple:
        """
        Find content matching a pattern with ...omit lines...
        Returns (matched_text, start_pos, end_pos) or None if not found.
        """
        m = self._match_placeholder(search_pattern)
        if not m:
            return None, None, None

        before = search_pattern[: m.start() - 1]
        after = search_pattern[m.end() + 1 :]

        # If either part is empty, handle accordingly
        if not before or not after:
            return None, None, None

        escaped_before = re.escape(before)
        escaped_after = re.escape(after)

        # Create a pattern that matches before...anything...after
        pattern = escaped_before + r".*?" + escaped_after
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return content[match.start() : match.end()], match.start(), match.end()

        return None, None, None

    def _fuzzy_replace(self, old_str: str, new_str: str, content: str) -> str | None:
        align = fuzz.partial_ratio_alignment(old_str, content)
        if align and align.score > 98:
            improved_range = self._improve_fuzz_match(content, old_str, align)
            if improved_range:
                start, end = improved_range
                return content[:start] + new_str + content[end:]

    def _improve_fuzz_match(
        self, content: str, old_str: str, align
    ) -> tuple[int, int] | None:
        content_lines = content.splitlines(keepends=True)
        line_starts = []
        curr = 0
        for line in content_lines:
            line_starts.append(curr)
            curr += len(line)
        line_starts.append(curr)

        # Align start and end of fuzzy matched old str to line boundary,
        # And try to find one candidate that matches all alnum sequence.
        dest_start, dest_end = align.dest_start, align.dest_end
        start_candidates = [0]
        end_candidates = [len(content)]
        for i in range(len(line_starts) - 1):
            current_idx = line_starts[i]
            next_idx = line_starts[i + 1]
            if current_idx <= dest_start and next_idx > dest_start:
                start_candidates = [current_idx, next_idx]
            if current_idx < dest_end and next_idx >= dest_end:
                end_candidates = [current_idx, next_idx]
                break

        def clean_str(sentence: str) -> str:
            string_out = _alnum_regex.sub("", sentence)
            return string_out.strip().lower()

        for s in start_candidates:
            for e in end_candidates:
                matched = content[s:e]
                if clean_str(old_str) == clean_str(matched):
                    return s, e

        return None


class Shell:
    def __init__(self, workspace_dir: str | Path):
        """
        Initialize the Shell tool.

        Args:
            workspace_dir: The absolute path to the directory that the shell is allowed to access (read/write).
                          On Linux, commands will be sandboxed using bwrap.
        """
        if not workspace_dir:
            raise ValueError("workspace_dir must be provided")

        self.workspace_dir = Path(workspace_dir)
        if not self.workspace_dir.is_absolute():
            raise ValueError(f"workspace_dir must be an absolute path: {workspace_dir}")
        self.disabled = False

        if sys.platform == "linux":
            self.bwrap_path = shutil.which("bwrap")
            if not self.bwrap_path:
                self.disabled = True
                logger.error("bwrap not found on linux, `Shell` tool disabled.")
        else:
            self.disabled = True
            logger.error("No sandbox implemented. `Shell` tool disabled.")

    def _get_sandboxed_cmd(self, command: str) -> list[str]:
        """Construct the bwrap command arguments."""
        if sys.platform == "linux":
            return self._get_linux_sandboxed_cmd(command)
        else:
            return []

    def _get_linux_sandboxed_cmd(self, command: str) -> list[str]:
        workspace_str = str(self.workspace_dir)
        home_dir = Path.home()
        home_str = str(home_dir)

        bwrap_args = [
            self.bwrap_path,
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/sbin",
            "/sbin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            home_str,
            home_str,
            "--bind",
            workspace_str,
            workspace_str,
        ]

        # Mask sensitive directories/files in home
        sensitive_paths = [
            ".ssh",
            ".gnupg",
        ]
        for p in sensitive_paths:
            full_path = home_dir / p
            if full_path.exists():
                bwrap_args.extend(["--tmpfs", str(full_path)])

        bwrap_args.extend(
            [
                "--chdir",
                workspace_str,
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
            ]
        )

        if os.path.exists("/lib64"):
            bwrap_args.extend(["--ro-bind", "/lib64", "/lib64"])

        # Essential files for networking and basic tools to work
        for path in [
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/passwd",
            "/etc/group",
            "/etc/ld.so.cache",
            "/etc/alternatives",
            "/etc/ssl",
            "/etc/ca-certificates",
        ]:
            if os.path.exists(path):
                bwrap_args.extend(["--ro-bind", path, path])

        bwrap_args.extend(["--", "/bin/bash", "-c", command])
        return bwrap_args

    @tool()
    async def shell(self, command: str, timeout: int | None = 100) -> str:
        """
        Run arbitrary shell commands in system's shell and return its output.

        Rules
            1. For searching code, use `rg` or `ast-grep`.
            2. Interactive commands that require user input are not supported and will fail.
            3. The command will be invoked by `bash -c`, mind the syntax. e.g.:
               - use single quote to avoid substution

        Examples
            command: "ls -la | rg staff"
            result: "total 24\\ndrwxr-xr-x  5 user  staff  160 Jan  1 12:00 .\\n..."

        Args:
            command: The shell command to execute (e.g., "ls -la", "pwd", "git status")
            timeout: Optional timeout in seconds for the command execution (default: 100)

        Returns:
            str: The combined stdout and stderr output of the command
        """
        try:
            logger.info(f"Executing shell command: {command}")
            sandboxed_cmd = self._get_sandboxed_cmd(command)

            env = os.environ.copy()

            process = await asyncio.create_subprocess_exec(
                *sandboxed_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                error_msg = f"Command timed out after {timeout} seconds"
                logger.error(error_msg)
                return error_msg

            # Combine stdout and stderr
            output = stdout.decode()
            stderr_output = stderr.decode()
            if stderr_output:
                if output:
                    output += "\n"
                output += stderr_output

            # Truncate output if it's too large
            lines = output.splitlines()
            truncated = truncate_content(lines)
            output = "\n".join(truncated["lines"])
            if truncated["truncated_by_bytes"] or truncated["has_more_lines"]:
                output += f"\n\n[Output truncated due to size limits. Total lines: {len(lines)}]"

            # Add return code information
            if process.returncode != 0:
                output += f"\n[Process exited with code {process.returncode}]"

            logger.info(f"Command completed with return code: {process.returncode}")
            return output

        except Exception as e:
            error_msg = f"Error executing command: {e!s}"
            logger.error(error_msg)
            return error_msg


class ProjectPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        self.project_manager = ProjectManager(agent)
        agent.register_dependency("project_manager", self.project_manager)
        self.file_edit = FileEdit()
        self.shell_tool = Shell(self.agent.workspace.absolute())

    @command(
        ["add", "add_file_list"],
        "Add files to context - /add <file1> [file2...] /add_file_list",
    )
    async def project_command(self, name: str, arg: str):
        project_manager = self.agent.get_dependency("project_manager")
        if name == "add":
            files = arg.split() if arg else []
            if not files:
                await self.agent.agent_io.agent_send("Please specify files.")
                return
            await project_manager.read_by_user(files)
        elif name == "add_file_list":
            project_manager.add_project_files()

    def get_completions(self, name, args):
        # Parse the arguments to get the current word being completed
        if not args:
            current_word = ""
        else:
            parts = args.split()
            if args.endswith(" "):
                current_word = ""
            else:
                current_word = parts[-1] if parts else ""

        if name == "add":
            project_manager = self.agent.get_dependency("project_manager")
            candidates = project_manager.candidates()
        else:
            candidates = []

        # Filter candidates based on current word
        for candidate in candidates:
            if current_word in candidate:
                yield Completion(
                    candidate, start_position=-len(current_word), display=candidate
                )

    def tools(self):
        tools = super().tools()
        tools.extend(
            [
                ToolDef(
                    func=self.project_manager.read,
                    kwargs=getattr(self.project_manager.read, "__tool_kwargs__", {}),
                ),
                ToolDef(
                    func=self.file_edit.replace_in_file,
                    kwargs=getattr(
                        self.file_edit.replace_in_file, "__tool_kwargs__", {}
                    ),
                ),
                ToolDef(
                    func=self.file_edit.write_to_file,
                    kwargs=getattr(self.file_edit.write_to_file, "__tool_kwargs__", {}),
                ),
            ]
        )
        if not self.shell_tool.disabled:
            tools.append(
                ToolDef(
                    func=self.shell_tool.shell,
                    kwargs=getattr(self.shell_tool.shell, "__tool_kwargs__", {}),
                )
            )
        return tools
