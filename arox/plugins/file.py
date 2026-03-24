import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completion
from pydantic_ai import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from rapidfuzz import fuzz

from arox.agent_patterns.plugin import Plugin, command, tool
from arox.plugins.capabilities import AGENT_INFO, AGENT_RESET, PROJECT_FILES
from arox.utils import DEFAULT_READ_LIMIT, truncate_content

if TYPE_CHECKING:
    from arox.agent_patterns.llm_base import LLMBaseAgent

logger = logging.getLogger(__name__)

_alnum_regex = re.compile(r"(?ui)\W")


class FilePlugin(Plugin):
    def __init__(self, agent: "LLMBaseAgent"):
        super().__init__(agent)
        self.workspace = agent.workspace
        self._pending_text_files: dict[str, str] = {}
        self._pending_binary_files: dict[str, bytes] = {}
        self.session_files = []

        self.agent.provide_capability(AGENT_INFO, self.get_info)
        self.agent.provide_capability(AGENT_RESET, self.reset)

        self.reset()

    def reset(self):
        self._pending_text_files = {}
        self._pending_binary_files = {}
        self.session_files = []

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

    def candidates(self):
        provided_files = []
        for get_files_func in self.agent.get_capability(PROJECT_FILES):
            files = get_files_func()
            if files:
                provided_files.extend(files)

        if provided_files:
            return provided_files

        # Fallback
        return [
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]

    def _normalize_path(self, file_path: str) -> Path:
        workspace = self.workspace
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

    def consume_pending(self) -> tuple[dict[str, str], dict[str, bytes]]:
        text_files = self._pending_text_files
        self._pending_text_files = {}

        binary_files = self._pending_binary_files
        self._pending_binary_files = {}

        return text_files, binary_files

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
                result["content"] = "\n".join(content_lines)

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
                chunk = f.read(4096)
                if not chunk:
                    return False

                if b"\0" in chunk:
                    return True

                non_printable = 0
                for b in chunk:
                    if b < 9 or (13 < b < 32):
                        non_printable += 1

                return non_printable / len(chunk) > 0.3
        except Exception:
            return True

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
        m = self._match_placeholder(search_pattern)
        if not m:
            return None, None, None

        before = search_pattern[: m.start() - 1]
        after = search_pattern[m.end() + 1 :]

        if not before or not after:
            return None, None, None

        escaped_before = re.escape(before)
        escaped_after = re.escape(after)

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

    @command(
        ["add"],
        "Add files to context - /add <file1> [file2...]",
    )
    async def file_command(self, name: str, arg: str):
        if name == "add":
            files = arg.split() if arg else []
            if not files:
                await self.agent.agent_io.agent_send("Please specify files.")
                return
            await self.read_by_user(files)

    def get_completions(self, name, args):
        if not args:
            current_word = ""
        else:
            parts = args.split()
            if args.endswith(" "):
                current_word = ""
            else:
                current_word = parts[-1] if parts else ""

        if name == "add":
            candidates = self.candidates()
        else:
            candidates = []

        for candidate in candidates:
            if current_word in candidate:
                yield Completion(
                    candidate, start_position=-len(current_word), display=candidate
                )

    async def get_info(self) -> str:
        session_files = self.session_files
        if session_files:
            info = f"\nChat files ({len(session_files)}):"
            for file_path in session_files:
                info += f"\n  - {file_path}"
            return info
        else:
            return "\nNo chat files currently loaded."

    async def history_processor(
        self, messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if messages and isinstance(messages[-1], ModelRequest):
            pending_text_files, pending_binary = self.consume_pending()

            extra_content = []

            for path, data in pending_binary.items():
                import mimetypes

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
                        "content": {content},
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
