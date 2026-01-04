from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from pydantic_ai import FunctionToolset

from arox import agent_patterns, commands
from arox.agent_patterns.chat import ChatAgent
from arox.config import TomlConfigParser
from arox.ui.io import TextIOAdapter
from arox.utils import user_input_generator


def multiply(a: int, b: int) -> int:
    """calculate a * b"""
    return a * b


@pytest.mark.asyncio
async def test_rewrite_agent():
    current_dir = Path(__file__).parent.absolute()
    default_agent_config = current_dir / "rewrite.toml"
    toml_parser = TomlConfigParser(
        config_files=[default_agent_config],
        override_configs={"workspace": str(current_dir)},
    )
    agent_patterns.init(toml_parser)
    file_name = Path(__file__).parent / "test_sample.md"
    test_user_msg = [
        f"/add {file_name}\n",
        "Translate file content to chinese and calculate 1488*2083.\n",
        f"/save {file_name}.testres\n",
        "\x04",
    ]
    local_toolset = FunctionToolset()
    local_toolset.add_function(multiply)
    with create_pipe_input() as pipe_input:

        async def user_input():
            return await user_input_generator(input=pipe_input)

        io_adapter = TextIOAdapter(user_input=user_input)
        agent = ChatAgent(
            "rewrite", toml_parser, io_adapter=io_adapter, toolsets=[local_toolset]
        )

        cmds = [commands.FileCommand(agent), commands.SaveCommand(agent)]
        agent.register_commands(cmds)

        for msg in test_user_msg:
            pipe_input.send_text(msg)
        # TODO move run_to_end to agent?
        await agent.io_channel.run_to_end(agent.start())
