import asyncio
import signal
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import ToolCallPart
from pydantic_ai.models.test import TestModel

from arox.core.app import app_setup
from arox.core.chat import ChatInputReply, ChatInputRequest, ChatServeDriver
from arox.core.plugin import CommandReply
from arox.core.runner import ServeRunner
from arox.core.session import AgentSession
from arox.ui.text_io import TextIOAdapter, UserInputGenerator


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


class FakeAgentIO:
    def __init__(self, inputs: list[str | None]):
        self.inputs = iter(inputs)
        self.events: list[object] = []

    async def send(self, event):
        self.events.append(event)
        if isinstance(event, ChatInputRequest):
            return ChatInputReply(
                req_id=event.req_id,
                input_content=next(self.inputs),
            )
        return None


@pytest.mark.asyncio
async def test_chat_driver_handles_slash_commands_before_starting_turn():
    agent_io = FakeAgentIO(["/info", None])
    command_manager = SimpleNamespace(
        try_handle_slash=AsyncMock(return_value=CommandReply(req_id="", output="info"))
    )
    runtime = SimpleNamespace(agent_io=agent_io, command_manager=command_manager)
    start_turn = AsyncMock()
    runner = cast(
        ServeRunner,
        SimpleNamespace(
            runtime=runtime,
            start_turn=start_turn,
            session=SimpleNamespace(record_turn_error=AsyncMock()),
        ),
    )

    await ChatServeDriver().serve(runner)

    command_manager.try_handle_slash.assert_awaited_once_with("/info")
    start_turn.assert_not_awaited()
    requests = [
        event for event in agent_io.events if isinstance(event, ChatInputRequest)
    ]
    assert len(requests) == 2
    assert "info" in agent_io.events


@pytest.mark.asyncio
async def test_chat_driver_sends_unknown_slash_commands_to_the_agent():
    agent_io = FakeAgentIO(["/unknown", None])
    command_manager = SimpleNamespace(try_handle_slash=AsyncMock(return_value=None))
    runtime = SimpleNamespace(agent_io=agent_io, command_manager=command_manager)
    start_turn = AsyncMock(return_value=SimpleNamespace(output="done"))
    runner = cast(
        ServeRunner,
        SimpleNamespace(
            runtime=runtime,
            start_turn=start_turn,
            session=SimpleNamespace(record_turn_error=AsyncMock()),
        ),
    )

    await ChatServeDriver().serve(runner)

    command_manager.try_handle_slash.assert_awaited_once_with("/unknown")
    start_turn.assert_awaited_once()
    await_args = start_turn.await_args
    assert await_args is not None
    reply = await_args.args[0]
    assert reply.text_content == "/unknown"


@pytest.mark.asyncio
async def test_text_sigint_only_invokes_bound_foreground_handler():
    io_adapter = TextIOAdapter()
    interrupted = asyncio.Event()

    async def interrupt_foreground():
        interrupted.set()

    io_adapter.set_interrupt_handler(interrupt_foreground)

    async with io_adapter:
        handler = cast(Callable[[int, Any], Any], signal.getsignal(signal.SIGINT))
        assert callable(handler)
        handler(signal.SIGINT, None)
        await asyncio.wait_for(interrupted.wait(), timeout=1)


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
        user_input = UserInputGenerator(input=pipe_input, output=DummyOutput())

        io_adapter = TextIOAdapter()
        session = AgentSession(path=["dummy"], agent_name="dummy_chat")
        io_adapter.user_input = user_input

        for msg in test_user_msg:
            pipe_input.send_text(msg)

        test_model = TestModel(call_tools=["multiply"])
        async with io_adapter:
            runner = ServeRunner(session, config_loader, io_adapter, ChatServeDriver())
            try:
                runtime = await runner.start()
                runtime.add_local_tool(multiply)
                with runtime._pydantic_agent.override(model=test_model):
                    runner.serve()
                    await runner.wait()
                    assert not session.is_active
            finally:
                await runner.stop()

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
