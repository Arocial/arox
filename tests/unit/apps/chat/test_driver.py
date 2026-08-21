from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import ToolCallPart
from pydantic_ai.models.test import TestModel

from arox.apps.chat.io_adapters.text import CommandCompleter, TextIOAdapter
from arox.core.agent_runtime import AgentRuntime
from arox.core.app import app_setup
from arox.core.plugin import CommandDispatchResult, CommandInput, CommandReply
from arox.core.session import AgentSession
from arox.core.types import UserInput


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


@pytest.mark.asyncio
async def test_runtime_handles_slash_commands_without_starting_turn():
    dispatch_command = AsyncMock()
    start_turn = Mock()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            _dispatch_command=dispatch_command,
            start_turn=start_turn,
        ),
    )
    result = await AgentRuntime.accept_input(runtime, UserInput(input_content="/info"))

    assert result is None
    dispatch_command.assert_awaited_once_with("/info")
    start_turn.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_starts_regular_user_input():
    turn = object()
    dispatch_command = AsyncMock()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            _dispatch_command=dispatch_command,
            start_turn=lambda event: turn,
        ),
    )
    event = UserInput(input_content="hello")
    assert await AgentRuntime.accept_input(runtime, event) is turn
    dispatch_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_handles_command_input_from_io():
    dispatch_command = AsyncMock(
        return_value=CommandDispatchResult(
            "handled", CommandReply(req_id="command", output="details")
        )
    )
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(_dispatch_command=dispatch_command),
    )
    command_input = CommandInput(command={"type": "InfoCommand"})

    reply = await AgentRuntime.accept_command(runtime, command_input)

    dispatch_command.assert_awaited_once_with({"type": "InfoCommand"})
    assert reply.req_id == command_input.req_id
    assert reply.status == "handled"
    assert reply.output == "details"


@pytest.mark.asyncio
async def test_text_adapter_keeps_one_user_input_per_channel():
    io_adapter = TextIOAdapter(output=DummyOutput())
    first_io = object()
    second_io = object()
    first_runtime = SimpleNamespace(
        command_manager=SimpleNamespace(completion_router=SimpleNamespace())
    )
    second_runtime = SimpleNamespace(
        command_manager=SimpleNamespace(completion_router=SimpleNamespace())
    )

    first = io_adapter._user_input_for(cast(Any, first_io), first_runtime)
    assert io_adapter._user_input_for(cast(Any, first_io), second_runtime) is first
    second = io_adapter._user_input_for(cast(Any, second_io), second_runtime)

    assert second is not first
    first_completer = first.session.completer
    second_completer = second.session.completer
    assert isinstance(first_completer, CommandCompleter)
    assert isinstance(second_completer, CommandCompleter)
    assert first_completer.runtime is first_runtime
    assert second_completer.runtime is second_runtime


@pytest.mark.asyncio
async def test_chat_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    # Create dummy config
    default_agent_config = tmp_path / ".arox" / "config.toml"
    default_agent_config.parent.mkdir(parents=True, exist_ok=True)
    default_agent_config.write_text("""
model_ref = "test"
[agent.dummy_chat]
system_prompt = "Hi there."
""")
    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    test_user_msg = [
        "Calculate 1488*2083.\n",
        "\x04",
    ]
    with create_pipe_input() as pipe_input:
        io_adapter = TextIOAdapter(input=pipe_input, output=DummyOutput())
        session = AgentSession(path=["dummy"], agent_name="dummy_chat")

        for msg in test_user_msg:
            pipe_input.send_text(msg)

        test_model = TestModel(call_tools=["multiply"])
        async with io_adapter:
            runtime = AgentRuntime(config_loader, io_adapter, session)
            async with runtime:
                runtime.add_local_tool(multiply)
                with runtime._pydantic_agent.override(model=test_model):
                    await runtime.agent_ep.wait_disconnected()
                    assert session.is_active
            assert not session.is_active

        # Verify that the tool was called
        messages = runtime.message_history
        from pydantic_ai.messages import ModelRequest, ModelResponse

        tool_calls = [
            part.tool_name
            for msg in messages
            if isinstance(msg, (ModelRequest, ModelResponse))
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert "multiply" in tool_calls
