import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


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
            return f"Error writing to file: {str(e)}"

    async def replace_in_file(self, path: str, old_str: str, new_str: str) -> str:
        """Searches for `old_str` in the file and replaces it with `new_str`.

        Args:
            path: The path of the file to modify.
            old_str: The block of code to be replaced.
                - It must be unique enough to identify the correct section.
                - It should include enough context (lines before and after) to ensure a correct match.
                - Prefer use "...omit lines..." on a line by itself to represent
                  uninterrupted code between a prefix and a suffix as long as the correct section
                  can be uniquely identified.
                Example:
                    old_str:
                        def my_func():
                            ...omit lines...
                            return True
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
                    align = fuzz.partial_ratio_alignment(old_str, content)
                    if align and align.score > 95:
                        content = (
                            content[: align.dest_start]
                            + new_str
                            + content[align.dest_end :]
                        )
                    else:
                        content = None

            if content:
                file_path.write_text(content)
                msg = f"Successfully updated {file_path}"
            else:
                msg = f"Cannot find a match for passed old_str in {file_path}"
            logger.info(msg)
            return msg
        except Exception as e:
            msg = f"Error replacing in file `{path}` with exception: {str(e)}"
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
