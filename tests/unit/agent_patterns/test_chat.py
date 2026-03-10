import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from pydantic_ai import FunctionToolset
from pydantic_ai.models.test import TestModel

from arox import agent_patterns
from arox.agent_patterns.chat import ChatAgent
from arox.config import TomlConfigParser
from arox.ui.io import TextIOAdapter
from arox.utils import user_input_generator


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


@pytest.mark.asyncio
async def test_chat_agent(tmp_path):
    # Create dummy config
    default_agent_config = tmp_path / "dummy_chat.toml"
    default_agent_config.write_text("""
[DEFAULT]
model_ref = "test"
[agent.dummy_chat]
system_prompt = "Hi there."
[agent.dummy_chat.model_params]
""")

    toml_parser = TomlConfigParser(
        config_files=[default_agent_config],
        override_configs={"workspace": str(tmp_path)},
    )
    agent_patterns.init(toml_parser)

    test_user_msg = [
        "Calculate 1488*2083.\n",
        "\x04",
    ]
    local_toolset = FunctionToolset()
    local_toolset.add_function(multiply)

    with create_pipe_input() as pipe_input:

        async def user_input():
            return await user_input_generator(input=pipe_input)

        from arox.ui.io import IOChannel

        io_channel = IOChannel()
        io_adapter = TextIOAdapter(io_channel)
        agent = ChatAgent(
            "dummy_chat", toml_parser, agent_io=io_channel, local_toolset=local_toolset
        )
        io_adapter.user_input = user_input

        for msg in test_user_msg:
            pipe_input.send_text(msg)

        asyncio.create_task(io_adapter.start())

        test_model = TestModel()
        with agent.pydantic_agent.override(model=test_model):
            async with agent:
                await agent.start()

        # Verify that the tool was called
        messages = agent.state.message_history
        tool_calls = [
            part.tool_name
            for msg in messages
            if hasattr(msg, "parts")
            for part in msg.parts
            if hasattr(part, "tool_name")
        ]
        assert "multiply" in tool_calls
