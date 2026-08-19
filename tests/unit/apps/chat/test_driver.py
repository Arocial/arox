import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import ToolCallPart
from pydantic_ai.models.test import TestModel

from arox.apps.chat.driver import ChatInputReply, ChatInputRequest, ChatServeDriver
from arox.apps.chat.io_adapters.text import CommandCompleter, TextIOAdapter
from arox.core.app import app_setup
from arox.core.plugin import CommandDispatchResult, CommandReply
from arox.core.runner import ServeRunner
from arox.core.session import AgentSession


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
    agent_ep = FakeAgentIO(["/info", None])
    command_manager = SimpleNamespace(
        dispatch=AsyncMock(
            return_value=CommandDispatchResult(
                "handled", CommandReply(req_id="", output="info")
            )
        )
    )
    run_turn = AsyncMock()
    runtime = SimpleNamespace(
        agent_ep=agent_ep,
        command_manager=command_manager,
        run_turn=run_turn,
        session=SimpleNamespace(id="chat"),
    )
    runner = cast(
        ServeRunner,
        SimpleNamespace(
            runtime=runtime,
            session=SimpleNamespace(record_error_event=AsyncMock()),
        ),
    )

    await ChatServeDriver().run(runner)

    command_manager.dispatch.assert_awaited_once_with("/info")
    run_turn.assert_not_awaited()
    requests = [
        event for event in agent_ep.events if isinstance(event, ChatInputRequest)
    ]
    assert len(requests) == 2
    assert "info" in agent_ep.events


@pytest.mark.asyncio
async def test_chat_driver_reports_unknown_slash_commands_without_starting_turn():
    agent_ep = FakeAgentIO(["/unknown", None])
    command_manager = SimpleNamespace(
        dispatch=AsyncMock(return_value=CommandDispatchResult("unknown"))
    )
    run_turn = AsyncMock(return_value=SimpleNamespace(output="done"))
    runtime = SimpleNamespace(
        agent_ep=agent_ep,
        command_manager=command_manager,
        run_turn=run_turn,
        session=SimpleNamespace(id="chat"),
    )
    runner = cast(
        ServeRunner,
        SimpleNamespace(
            runtime=runtime,
            session=SimpleNamespace(record_error_event=AsyncMock()),
        ),
    )

    await ChatServeDriver().run(runner)

    command_manager.dispatch.assert_awaited_once_with("/unknown")
    run_turn.assert_not_awaited()
    assert "Unknown command." in agent_ep.events


@pytest.mark.asyncio
async def test_chat_driver_keeps_serving_after_interaction_exception():
    agent_ep = FakeAgentIO(["hello", None])
    error = ValueError("bad response")
    runtime = SimpleNamespace(
        agent_ep=agent_ep,
        command_manager=SimpleNamespace(dispatch=AsyncMock()),
        run_turn=AsyncMock(side_effect=error),
        session=SimpleNamespace(id="chat", agent_name="coder"),
    )
    runner = cast(ServeRunner, SimpleNamespace(runtime=runtime))

    await ChatServeDriver().run(runner)

    assert sum(isinstance(event, ChatInputRequest) for event in agent_ep.events) == 2


@pytest.mark.asyncio
async def test_chat_driver_cancels_current_interaction_and_keeps_serving():
    agent_ep = FakeAgentIO(["hello", None])
    command_manager = SimpleNamespace(dispatch=AsyncMock())
    started = asyncio.Event()

    async def blocking_turn(user_input):
        started.set()
        await asyncio.Event().wait()

    runtime = SimpleNamespace(
        agent_ep=agent_ep,
        command_manager=command_manager,
        run_turn=blocking_turn,
        session=SimpleNamespace(id="chat"),
    )
    runner = cast(
        ServeRunner,
        SimpleNamespace(
            runtime=runtime,
            session=SimpleNamespace(record_error_event=AsyncMock()),
        ),
    )
    driver = ChatServeDriver()
    serve_task = asyncio.create_task(driver.run(runner))

    await started.wait()
    assert await driver.cancel_current_execution()
    await serve_task

    assert not await driver.cancel_current_execution()


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
            runner = ServeRunner(session, config_loader, io_adapter, ChatServeDriver())
            async with runner:
                runtime = runner.runtime
                assert runtime is not None
                runtime.add_local_tool(multiply)
                with runtime._pydantic_agent.override(model=test_model):
                    await runner.run()
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
