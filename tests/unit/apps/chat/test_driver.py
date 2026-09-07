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
from arox.core.session import AgentSession
from arox.core.types import (
    ClientInput,
    CommandPayload,
    MessagePayload,
    normalize_client_input,
)


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


@pytest.mark.asyncio
async def test_runtime_handles_slash_commands_without_starting_turn():
    start_turn = Mock()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            _run_command=AsyncMock(),
            _command_tasks=set(),
            _active_command_ids=set(),
            _checkpoint_if_idle=Mock(),
            session=SimpleNamespace(id="session"),
            agent_ep=SimpleNamespace(send=AsyncMock()),
            start_turn=start_turn,
        ),
    )
    client_input = normalize_client_input(
        ClientInput(payload=MessagePayload(content="/info"))
    )
    result = await AgentRuntime.accept_input(runtime, client_input)

    assert result is client_input
    assert isinstance(client_input.payload, CommandPayload)
    assert client_input.payload.status == "accepted"
    start_turn.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_starts_regular_user_input():
    turn = object()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            start_turn=lambda event: turn,
        ),
    )
    event = normalize_client_input(ClientInput(payload=MessagePayload(content="hello")))
    result = await AgentRuntime.accept_input(runtime, event)
    assert result is event
    assert isinstance(event.payload, MessagePayload)
    assert event.payload.status is None


@pytest.mark.asyncio
async def test_runtime_accepts_structured_command_input():
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            _run_command=AsyncMock(),
            _command_tasks=set(),
            _active_command_ids=set(),
            _checkpoint_if_idle=Mock(),
            session=SimpleNamespace(id="session"),
            agent_ep=SimpleNamespace(send=AsyncMock()),
        ),
    )
    structured_input = ClientInput(
        payload=CommandPayload(command={"type": "InfoCommand"})
    )

    result = await AgentRuntime.accept_input(runtime, structured_input)

    assert result is structured_input
    assert isinstance(result.payload, CommandPayload)
    assert result.payload.status == "accepted"


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
        messages = runtime.session.message_history
        from pydantic_ai.messages import ModelRequest, ModelResponse

        tool_calls = [
            part.tool_name
            for msg in messages
            if isinstance(msg, (ModelRequest, ModelResponse))
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert "multiply" in tool_calls
