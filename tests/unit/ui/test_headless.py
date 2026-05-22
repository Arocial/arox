import pytest
from pydantic_ai import FunctionToolset
from pydantic_ai.models.test import TestModel

from arox.core.app import app_setup
from arox.core.chat import ChatAgent
from arox.ui.headless import HeadlessIOAdapter


@pytest.mark.asyncio
async def test_headless_runs_one_step_and_exits(tmp_path, capsys):
    default_agent_config = tmp_path / "dummy_chat.toml"
    default_agent_config.write_text("""
model_ref = "test"
[agent.dummy_chat]
system_prompt = "Hi there."
""")

    parsed_config = app_setup(
        config_files=[default_agent_config],
        cli_args={"workspace": str(tmp_path)},
    )

    from arox.core.llm_base import AgentDeps

    local_toolset = FunctionToolset[AgentDeps]()

    io_adapter = HeadlessIOAdapter(prompt="say hello")
    agent = ChatAgent(
        "dummy_chat",
        parsed_config,
        io_adapter=io_adapter,
        local_toolset=local_toolset,
    )

    test_model = TestModel(custom_output_text="hello world")
    with agent.pydantic_agent.override(model=test_model):
        async with io_adapter, agent:
            await agent.run()

    captured = capsys.readouterr()
    assert "hello world" in captured.out
    assert io_adapter.error is None


@pytest.mark.asyncio
async def test_headless_records_step_exception(tmp_path):
    default_agent_config = tmp_path / "dummy_chat.toml"
    default_agent_config.write_text("""
model_ref = "test"
[agent.dummy_chat]
system_prompt = "Hi there."
""")

    parsed_config = app_setup(
        config_files=[default_agent_config],
        cli_args={"workspace": str(tmp_path)},
    )

    from arox.core.llm_base import AgentDeps

    local_toolset = FunctionToolset[AgentDeps]()
    io_adapter = HeadlessIOAdapter(prompt="boom")
    agent = ChatAgent(
        "dummy_chat",
        parsed_config,
        io_adapter=io_adapter,
        local_toolset=local_toolset,
    )

    async def failing_step(*args, **kwargs):
        raise RuntimeError("step blew up")

    agent.step = failing_step  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    async with io_adapter, agent:
        await agent.run()

    assert io_adapter.error is not None
    assert "step blew up" in str(io_adapter.error)
