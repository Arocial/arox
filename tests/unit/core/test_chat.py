from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import ToolCallPart
from pydantic_ai.models.test import TestModel

from arox.core.app import app_setup
from arox.core.chat import ChatServeDriver
from arox.core.runner import ServingRunner
from arox.core.session import AgentSession
from arox.ui.text_io import TextIOAdapter, UserInputGenerator


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


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
            runner = ServingRunner(
                session, config_loader, io_adapter, ChatServeDriver()
            )
            try:
                agent = await runner.start()
                agent.add_local_tool(multiply)
                with agent.pydantic_agent.override(model=test_model):
                    runner.serve()
                    await runner.wait()
                    assert not session.is_active
            finally:
                await runner.stop()

        # Verify that the tool was called
        messages = agent.message_history
        from pydantic_ai.messages import ModelRequest, ModelResponse

        tool_calls = [
            part.tool_name
            for msg in messages
            if isinstance(msg, (ModelRequest, ModelResponse))
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert "multiply" in tool_calls
