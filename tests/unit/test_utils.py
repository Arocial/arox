import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from arox.utils import (
    deep_merge,
    render_template,
    truncate_content,
    user_input_generator,
)


def test_deep_merge_basic():
    source = {"a": 1, "b": {"c": 2}}
    overrides = {"b": {"c": 3, "d": 4}, "e": 5}
    result = deep_merge(source, overrides)
    assert result == {"a": 1, "b": {"c": 3, "d": 4}, "e": 5}


def test_deep_merge_empty_source():
    source = {}
    overrides = {"a": 1, "b": {"c": 2}}
    result = deep_merge(source, overrides)
    assert result == {"a": 1, "b": {"c": 2}}


def test_deep_merge_empty_overrides():
    source = {"a": 1, "b": {"c": 2}}
    overrides = {}
    result = deep_merge(source, overrides)
    assert result == {"a": 1, "b": {"c": 2}}


def test_deep_merge_nested_structures():
    source = {"a": {"b": {"c": 1}}, "d": [1, 2]}
    overrides = {"a": {"b": {"c": 2}}, "d": [3, 4]}
    result = deep_merge(source, overrides)
    assert result == {"a": {"b": {"c": 2}}, "d": [3, 4]}


@pytest.mark.asyncio
async def test_user_input_generator_quit():
    with create_pipe_input() as pipe_input:
        pipe_input.send_text("test1\n")

        result = await user_input_generator(
            input=pipe_input,
            output=DummyOutput(),
        )
        assert result == "test1"

        pipe_input.send_text("\x04")  # EOF (Ctrl+D)
        with pytest.raises(EOFError):
            await user_input_generator(
                input=pipe_input,
                output=DummyOutput(),
            )


def test_render_template():
    template = "Hello {{ name }}!"
    result = render_template(template, name="World")
    assert result == "Hello World!"


def test_truncate_content_basic():
    lines = ["line 1", "line 2", "line 3"]
    result = truncate_content(lines)
    assert result["lines"] == ["line 1", "line 2", "line 3"]
    assert result["last_read_line"] == 3
    assert result["truncated_by_bytes"] is False
    assert result["has_more_lines"] is False
    assert result["offset"] == 0


def test_truncate_content_limit():
    lines = ["line 1", "line 2", "line 3"]
    result = truncate_content(lines, limit=2)
    assert result["lines"] == ["line 1", "line 2"]
    assert result["last_read_line"] == 2
    assert result["truncated_by_bytes"] is False
    assert result["has_more_lines"] is True
    assert result["offset"] == 0


def test_truncate_content_offset():
    lines = ["line 1", "line 2", "line 3"]
    result = truncate_content(lines, offset=1)
    assert result["lines"] == ["line 2", "line 3"]
    assert result["last_read_line"] == 3
    assert result["truncated_by_bytes"] is False
    assert result["has_more_lines"] is False
    assert result["offset"] == 1


def test_truncate_content_max_line_length():
    lines = ["this is a very long line", "short line"]
    result = truncate_content(lines, max_line_length=10)
    assert result["lines"] == ["this is a ...", "short line"]
    assert result["last_read_line"] == 2
    assert result["truncated_by_bytes"] is False
    assert result["has_more_lines"] is False


def test_truncate_content_max_bytes():
    lines = ["line 1", "line 2", "line 3"]
    result = truncate_content(lines, max_bytes=10)
    assert result["lines"] == ["line 1"]
    assert result["last_read_line"] == 1
    assert result["truncated_by_bytes"] is True
    assert result["has_more_lines"] is True
