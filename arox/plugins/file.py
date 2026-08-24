import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic_ai import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    RunContext,
)
from rapidfuzz import fuzz

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.message_utils import internal_user_prompt_part
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, tool
from arox.plugins.slots import (
    AGENT_INFO,
    PERSISTENT_CONTEXT,
    PROJECT_FILES,
)
from arox.utils import DEFAULT_READ_LIMIT, truncate_content

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)

_whitespace_regex = re.compile(r"\s+")

# When set, failed `replace_in_file` matches are dumped to this directory
# (a copy of the target file plus the old/new strings) for later debugging.
REPLACE_DEBUG_DIR_ENV = "AROX_REPLACE_DEBUG_DIR"

MAX_BINARY_READ_BYTES = 1 * 1024 * 1024


@dataclass(kw_only=True)
class FileAddEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("add",)
    description: ClassVar[str] = "Add files to context - /add <file1> [file2...]"

    files: list[str]

    @classmethod
    def from_slash(cls, name, arg):
        return cls(files=arg.split() if arg else [])


class FilePlugin(Plugin):
    def __init__(self, runtime: "AgentRuntime"):
        super().__init__(runtime)
        self.workspace = runtime.workspace
        self._pending_text_files: dict[str, str] = {}
        self._pending_binary_files: dict[str, bytes] = {}
        self.session_files = []
        self.persistent_files: dict[str, str] = {}

        self._initialize_context()

    def on_load(self):
        self.runtime.provide_slot(AGENT_INFO, self.get_info)
        self.runtime.provide_slot(PERSISTENT_CONTEXT, self.get_persistent_context)

    def _initialize_context(self):
        self._pending_text_files = {}
        self._pending_binary_files = {}
        self.session_files = []
        self.persistent_files = {}

        # Auto read AGENTS.override.md or AGENTS.md if present
        for name in ("AGENTS.override.md", "AGENTS.md"):
            item = self.workspace / name
            if item.is_file():
                try:
                    content = "".join(self._read_text(name))
                    if not self.runtime.session.initialized:
                        self._pending_text_files[name] = content
                    self.persistent_files[name] = content
                    self._add_to_session(name)
                    break
                except Exception:
                    pass

    def get_persistent_context(self) -> list[ModelMessage]:
        if not self.persistent_files:
            return []

        text = "The following files are provided for reference:\n\n"
        for path, content in self.persistent_files.items():
            text += f'<file path="{path}">\n{content}\n</file>\n\n'

        return [ModelRequest(parts=[internal_user_prompt_part(text.strip())])]

    async def candidates(self):
        provided_files = []
        for files in await self.runtime.invoke_slot(PROJECT_FILES) or []:
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

    def _normalize_path(self, file_path: Path | str) -> Path:
        workspace = self.workspace
        p = Path(file_path)
        if not p.is_absolute():
            p = (workspace / p).absolute()
        return p

    def _add_to_session(self, file_path: Path | str):
        file_path = self._normalize_path(file_path)
        if file_path not in self.session_files:
            self.session_files.append(file_path)

    def _read_text(self, file_path: str) -> list[str]:
        path = self._normalize_path(file_path)
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
                    lines = self._read_text(file_path)
                    self._pending_text_files[file_path] = "".join(lines)
                self._add_to_session(path)
            except Exception as e:
                await self.runtime.agent_ep.send(
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
    ) -> str | BinaryContent:
        """Reads a file from the local filesystem.
        It's better to read multiple files as a batch that are potentially useful.

        Supports text files (with offset/limit line slicing) and binary files
        such as images or PDFs (returned as BinaryContent; offset/limit ignored).

        Args:
            path: The path to the file to read.
            offset: The line number to start reading from (0-based). Text only.
            limit: The number of lines to read (defaults to 2000). Text only.
        """
        try:
            normalized = self._normalize_path(path)
            if self._is_binary_file(normalized):
                import mimetypes

                size = normalized.stat().st_size
                if size > MAX_BINARY_READ_BYTES:
                    return (
                        f"Error: binary file {path} is {size} bytes, "
                        f"exceeds limit of {MAX_BINARY_READ_BYTES} bytes."
                    )
                with open(normalized, "rb") as f:
                    data = f.read()
                mime_type, _ = mimetypes.guess_type(str(normalized))
                if not mime_type:
                    mime_type = "application/octet-stream"
                self._add_to_session(path)
                return BinaryContent(data=data, media_type=mime_type)

            lines = self._read_text(path)

            truncated = truncate_content(lines, offset, limit)
            content_lines = truncated["lines"]
            last_read_line = truncated["last_read_line"]

            result = ""
            if content_lines:
                result = "\n".join(content_lines)

            if truncated["truncated_by_bytes"]:
                result += (
                    f"\n\n[Output truncated at {truncated['max_bytes']} bytes. "
                    f"Use 'offset' parameter to read beyond line {last_read_line}]"
                )
            elif truncated["has_more_lines"]:
                result += (
                    f"\n\n[File has more lines. "
                    f"Use 'offset' parameter to read beyond line {last_read_line}]"
                )

            self._add_to_session(path)
            return result

        except Exception as e:
            logger.error(f"Error reading file {path}: {e!s}")
            return f"Error reading file: {e!s}"

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
            file_path = self._normalize_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing to file: {e!s}"

    @tool(sequential=True)
    async def replace_in_file(self, path: str, old_str: str, new_str: str) -> str:
        """Searches for `old_str` in the file and replaces it with `new_str`.
        If you need to make multiple replacements in one or more files, please call this tool multiple times in a single response. The tool calls will be executed sequentially in the order they are provided.

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
            file_path = self._normalize_path(path)
            if not file_path.exists():
                return f"File not found: {file_path}"

            orig_content = file_path.read_text()

            new_content, status = self._resolve_replacement(
                orig_content, old_str, new_str
            )
            if status == "ok" and new_content is not None:
                file_path.write_text(new_content)
                msg = f"Successfully updated {file_path}"
            elif status == "ambiguous":
                msg = (
                    f"old_str matches multiple locations in {file_path}. "
                    "Make it unique by including more surrounding context."
                )
                self._dump_failed_replace(file_path, old_str, new_str)
            else:
                msg = (
                    f"Cannot find a match for passed old_str in {file_path}. "
                    "Please use the `read` tool to read the file again and ensure your `old_str` matches the file exactly."
                )
                self._dump_failed_replace(file_path, old_str, new_str)
            logger.info(msg)
            return msg
        except Exception as e:
            msg = f"Error replacing in file `{path}` with exception: {e!s}"
            logger.info(msg)
            return msg

    def _resolve_replacement(
        self, content: str, old_str: str, new_str: str
    ) -> tuple[str | None, str]:
        """Locate `old_str` and produce the replaced content.

        Returns ``(new_content, status)`` where status is ``"ok"``,
        ``"ambiguous"`` or ``"not_found"``. Every path requires `old_str` to
        identify exactly one location; matching multiple places yields
        ``"ambiguous"`` (we refuse rather than guess which one to edit).
        """
        # 1. Exact substring match.
        count = content.count(old_str)
        if count > 1:
            return None, "ambiguous"
        if count == 1:
            return content.replace(old_str, new_str, 1), "ok"

        # 2. Whitespace-tolerant fuzzy match.
        return self._fuzzy_replace(old_str, new_str, content)

    def _dump_failed_replace(self, file_path: Path, old_str: str, new_str: str) -> None:
        """Dump a failed `replace_in_file` match to a debug directory.

        Enabled by setting the ``AROX_REPLACE_DEBUG_DIR`` environment variable.
        Writes a copy of the target file along with the old/new strings into a
        timestamped subdirectory so the mismatch can be inspected later.
        """
        debug_dir = os.environ.get(REPLACE_DEBUG_DIR_ENV)
        if not debug_dir:
            return
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            dump_dir = Path(debug_dir).expanduser() / f"{file_path.name}-{stamp}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dump_dir / file_path.name)
            (dump_dir / "old_str.txt").write_text(old_str)
            (dump_dir / "new_str.txt").write_text(new_str)
            (dump_dir / "meta.txt").write_text(f"original_path: {file_path}\n")
            logger.info("Dumped failed replace_in_file context to %s", dump_dir)
        except Exception as e:
            logger.warning("Failed to dump replace_in_file debug context: %s", e)

    @staticmethod
    def _strip_lines(s: str) -> str:
        """Strip leading/trailing whitespace from each line, preserving line count.

        Makes fuzzy matching insensitive to indentation and trailing-space
        differences. Every line keeps a trailing ``\\n`` so that line offsets in
        the result stay in 1:1 correspondence with the original text's lines.
        """
        return "".join(line.strip() + "\n" for line in s.splitlines(keepends=True))

    def _fuzzy_replace(
        self, old_str: str, new_str: str, content: str
    ) -> tuple[str | None, str]:
        # Bail out on degenerate needles: a search string that is empty or only
        # whitespace would otherwise match almost anywhere after stripping.
        cleaned_old = _whitespace_regex.sub("", old_str)
        if not cleaned_old:
            return None, "not_found"

        match_range = self._find_fuzzy_range(old_str, content)
        if match_range is None:
            return None, "not_found"

        start, end = match_range
        # Ambiguity check: drop the matched lines and try again. The matched
        # range is line-aligned, so removing it leaves clean line boundaries.
        # A second fuzzy match means `old_str` is not unique and we refuse to
        # guess which location to edit.
        remainder = content[:start] + content[end:]
        if self._find_fuzzy_range(old_str, remainder) is not None:
            return None, "ambiguous"

        return content[:start] + new_str + content[end:], "ok"

    def _find_fuzzy_range(self, old_str: str, content: str) -> tuple[int, int] | None:
        """Locate a single whitespace-tolerant match of `old_str` in `content`.

        Returns the ``(start, end)`` character range of the match, or ``None``
        when no sufficiently good match is found.
        """
        align = fuzz.partial_ratio_alignment(
            self._strip_lines(old_str), self._strip_lines(content)
        )
        if align and align.score > 98:
            return self._improve_fuzz_match(content, old_str, align)
        return None

    def _improve_fuzz_match(
        self, content: str, old_str: str, align
    ) -> tuple[int, int] | None:
        # ``align`` offsets index into the line-stripped form of content (see
        # _fuzzy_replace), while the replacement slices the original content.
        # Both line-start tables are derived from the same line list, so a line
        # index is valid in either one.
        content_lines = content.splitlines(keepends=True)
        content_starts: list[int] = []
        stripped_starts: list[int] = []
        c_curr = s_curr = 0
        for line in content_lines:
            content_starts.append(c_curr)
            stripped_starts.append(s_curr)
            c_curr += len(line)
            s_curr += len(line.strip()) + 1  # +1 for the "\n" added by _strip_lines
        content_starts.append(c_curr)
        stripped_starts.append(s_curr)

        def line_index(starts: list[int], pos: int) -> int:
            for i in range(len(starts) - 1):
                if starts[i] <= pos < starts[i + 1]:
                    return i
            return len(starts) - 1

        def neighbors(idx: int, offsets: tuple[int, ...]) -> list[int]:
            seen: set[int] = set()
            for d in offsets:
                j = max(0, min(len(content_starts) - 1, idx + d))
                seen.add(content_starts[j])
            return sorted(seen)

        dest_start, dest_end = align.dest_start, align.dest_end
        start_line = line_index(stripped_starts, dest_start)
        # dest_end is exclusive; subtract 1 to find the line it sits inside.
        end_line = line_index(stripped_starts, max(dest_start, dest_end - 1))
        start_candidates = neighbors(start_line, (-1, 0, 1))
        end_candidates = neighbors(end_line, (0, 1, 2))

        def clean_str(sentence: str) -> str:
            return _whitespace_regex.sub("", sentence)

        cleaned_old = clean_str(old_str)
        for s in start_candidates:
            for e in end_candidates:
                if s >= e:
                    continue
                if clean_str(content[s:e]) == cleaned_old:
                    return s, e

        return None

    def commands(self):
        return [CommandSpec(FileAddEvent, self.handle_file_add, self.get_completions)]

    async def handle_file_add(self, event: "FileAddEvent"):
        if not event.files:
            await self.runtime.agent_ep.send("Please specify files.")
            return
        await self.read_by_user(event.files)

    async def get_completions(self, req: CompletionRequest):
        current_word = req.current_token
        for candidate in await self.candidates():
            if current_word in candidate:
                yield CompletionItem(
                    value=candidate,
                    label=candidate,
                    group="add",
                )

    async def get_info(self) -> str:
        session_files = self.session_files
        workspace = self.workspace
        if session_files:
            info = f"\nChat files ({len(session_files)}):"
            for file_path in session_files:
                if file_path.is_relative_to(workspace):
                    file_path = file_path.relative_to(workspace)
                info += f"\n  - {file_path}"
            return info
        else:
            return "\nNo chat files currently loaded."

    async def history_processor(
        self, ctx: RunContext[Any], messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if messages and isinstance(messages[-1], ModelRequest):
            pending_text_files, pending_binary = self.consume_pending()

            extra_content: list[str | BinaryContent] = []

            for path, data in pending_binary.items():
                import mimetypes

                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "application/octet-stream"
                extra_content.append(BinaryContent(data=data, media_type=mime_type))

            if pending_text_files:
                text_part = "The following files are provided for reference:\n\n"
                for path, text in pending_text_files.items():
                    text_part += f'<file path="{path}">\n{text}\n</file>\n\n'
                extra_content.append(text_part.strip())

            if extra_content:
                new_request = ModelRequest(
                    parts=[internal_user_prompt_part(extra_content)]
                )
                messages.insert(-1, new_request)

        return messages
