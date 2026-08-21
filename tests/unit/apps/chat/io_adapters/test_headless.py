from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from arox.apps.chat.io_adapters.headless import HeadlessIOAdapter
from arox.core.agent_runtime import AgentRuntime
from arox.core.app import app_setup
from arox.core.session import AgentSession


@pytest.mark.asyncio
async def test_headless_runs_one_step_and_exits(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
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

    io_adapter = HeadlessIOAdapter()
    session = AgentSession(path=["dummy"], agent_name="dummy_chat")

    test_model = TestModel(custom_output_text="hello world")
    async with io_adapter:
        runtime = AgentRuntime(config_loader, io_adapter, session)
        async with runtime:
            with runtime._pydantic_agent.override(model=test_model):
                turn = await runtime.accept_input("say hello")
                assert turn is not None
                await turn

    captured = capsys.readouterr()
    assert captured.out.strip()
    assert io_adapter.error is None


@pytest.mark.asyncio
async def test_headless_records_step_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
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

    io_adapter = HeadlessIOAdapter()
    session = AgentSession(path=["dummy"], agent_name="dummy_chat")

    async def failing_inference(*args, **kwargs):
        return SimpleNamespace(
            output=RuntimeError("step blew up"),
            all_messages=lambda: [],
        )

    async with io_adapter:
        runtime = AgentRuntime(config_loader, io_adapter, session)
        async with runtime:
            runtime._run_inference = failing_inference  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
            with pytest.raises(RuntimeError, match="step blew up"):
                turn = await runtime.accept_input("boom")
                assert turn is not None
                await turn

    assert io_adapter.error is not None
    assert "step blew up" in str(io_adapter.error)
