import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


_alnum_regex = re.compile(r"(?ui)\W")


class FileEdit:
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
