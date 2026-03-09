from typing import Any

import yaml
from jinja2 import Template
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024


def deep_merge(source, overrides):
    """Deep merge two dictionaries, with overrides taking precedence"""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source:
            source[key] = deep_merge(source.get(key, {}), value)
        else:
            source[key] = value
    return source


def parse_dict(value: str) -> dict:
    """Parse a string of key=value pairs into a dictionary"""
    if not value.strip():
        return {}
    return dict(yaml.safe_load(value))


def render_template(template, **kwargs):
    template = Template(template)
    return template.render(**kwargs)


async def user_input_generator(completer=None, input=None, output=None):
    """Async generator that yields user input"""
    history = FileHistory(".arox_history")
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):  # Enter to submit
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter newline
    @kb.add("escape", "O", "M")  # Shift+Enter (at least in my konsole)
    def _(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        prompt_continuation="> ",
        multiline=True,
        key_bindings=kb,
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        mouse_support=False,
        completer=completer,
        input=input,
        output=output,
    )
    return await session.prompt_async("\nUser (Ctrl+D to quit): ")


def xml_wrap(contents: list[tuple[str, str]]) -> str:
    xmled = []
    for tag, content in contents:
        if content is not None:
            xmled.append(f"<{tag}>\n{content}\n</{tag}>\n")
    return "\n".join(xmled)


async def run_command(command: str) -> tuple[str, str, int]:
    """
    Run a shell command asynchronously.

    Args:
        command: The command to run

    Returns:
        tuple: (stdout, stderr, return_code)
    """
    import asyncio

    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


def truncate_content(
    lines: list[str],
    offset: int = 0,
    limit: int = DEFAULT_READ_LIMIT,
    max_line_length: int = MAX_LINE_LENGTH,
    max_bytes: int = MAX_BYTES,
) -> dict[str, Any]:
    """
    Truncate a list of lines based on line count and byte size.

    Args:
        lines: List of strings (lines of the file)
        offset: Starting line index
        limit: Maximum number of lines to read
        max_line_length: Maximum length of a single line
        max_bytes: Maximum total bytes for the output

    Returns:
        dict: A dictionary containing:
            - lines: List of truncated strings
            - offset: The actual starting offset (by line) used
            - last_read_line: Index of the last line read
            - has_more_lines: Whether there are more lines in the file
            - limit: The line limit used
            - truncated_by_bytes: Whether truncation occurred due to byte limit
            - max_bytes: The maximum byte limit used
            - max_line_length: The maximum line length used
    """
    total_lines = len(lines)
    # Clamp offset and calculate end
    actual_offset = max(0, min(offset, total_lines))
    end = min(total_lines, actual_offset + limit)

    raw = []
    bytes_count = 0
    truncated_by_bytes = False
    last_read_line = actual_offset

    for i in range(actual_offset, end):
        line = lines[i].rstrip("\n\r")
        if len(line) > max_line_length:
            line = line[:max_line_length] + "..."

        # Estimate bytes for the line + newline
        line_bytes = len(line.encode("utf-8")) + (1 if raw else 0)
        if bytes_count + line_bytes > max_bytes:
            truncated_by_bytes = True
            break

        raw.append(line)
        bytes_count += line_bytes
        last_read_line = i + 1

    has_more_lines = total_lines > last_read_line
    return {
        "lines": raw,
        "last_read_line": last_read_line,
        "truncated_by_bytes": truncated_by_bytes,
        "has_more_lines": has_more_lines,
        "offset": actual_offset,
        "limit": limit,
        "max_line_length": max_line_length,
        "max_bytes": max_bytes,
    }
