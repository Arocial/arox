from unittest.mock import patch

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from arox.utils import deep_merge, run_command, user_input_generator


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


@pytest.mark.asyncio
async def test_run_command_success():
    """Test run_command with a successful command."""
    with patch("asyncio.create_subprocess_shell") as mock_create:
        mock_process = mock_create.return_value
        mock_process.communicate.return_value = (b"stdout", b"stderr")
        mock_process.returncode = 0

        stdout, stderr, returncode = await run_command("echo hello")
        assert stdout == "stdout"
        assert stderr == "stderr"
        assert returncode == 0


@pytest.mark.asyncio
async def test_run_command_failure():
    """Test run_command with a failing command."""
    with patch("asyncio.create_subprocess_shell") as mock_create:
        mock_process = mock_create.return_value
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1

        stdout, stderr, returncode = await run_command("false")
        assert stdout == ""
        assert stderr == "error"
        assert returncode == 1
