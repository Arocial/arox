import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    TextContent,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from arox.core.app import app_setup
from arox.core.io import AbstractIOAdapter, RequestEvent
from arox.core.llm_base import LLMBaseAgent
from arox.core.runner import TaskRunner
from arox.core.session import AgentSession
from arox.plugins.core import SetModelEvent


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_io, event):
        pass


@asynccontextmanager
async def _managed_runtime(agent, config_loader, io_adapter):
    runner = SimpleNamespace(runtime=agent)
    agent.session.runner = runner
    try:
        async with agent:
            yield agent
    finally:
        agent.session.runner = None


@pytest.mark.asyncio
async def test_agent_skills_filtering(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = ["skill1"]
""")

    # Monkeypatch Path.cwd to return tmp_path so ConfigLoader finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        assert "skill1" in agent.skill_catalog
    assert "skill2" not in agent.skill_catalog


@pytest.mark.asyncio
async def test_agent_skills_string(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = "skill2"
""")

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        assert "skill1" not in agent.skill_catalog
    assert "skill2" in agent.skill_catalog


@pytest.mark.asyncio
async def test_agent_skills_none(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
""")

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        assert "skill1" in agent.skill_catalog
    assert "skill2" in agent.skill_catalog


@pytest.mark.asyncio
async def test_request_event_dispatches_to_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    class CustomEvent(RequestEvent):
        pass

    received: list[RequestEvent] = []

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )

    async def handler(event):
        received.append(event)

    agent.register_request_handler(CustomEvent, handler)

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        ev = CustomEvent()
        await agent.adapter_io.send(ev)

    assert received == [ev]


@pytest.mark.asyncio
async def test_set_model_event_updates_model_ref(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )

    calls: list[str] = []
    original_set_model = agent.set_model

    def spy(ref):
        calls.append(ref)
        original_set_model(ref)

    agent.set_model = spy  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        calls.clear()
        agent.register_request_handler(
            SetModelEvent, lambda e: agent.set_model(e.model_ref)
        )
        await agent.adapter_io.send(SetModelEvent(model_ref="test"))

    assert calls == ["test"]
    assert agent.model_ref == "test"


def test_build_skill_catalog():
    assert LLMBaseAgent._build_skill_catalog({}) == ""

    skills = {
        "test_skill": {
            "name": "test_skill",
            "description": "A test skill",
            "location": "/path/to/SKILL.md",
        }
    }

    catalog = LLMBaseAgent._build_skill_catalog(skills)
    assert "<available_skills>" in catalog
    assert "<name>test_skill</name>" in catalog
    assert "<description>A test skill</description>" in catalog
    assert "<location>/path/to/SKILL.md</location>" in catalog


@pytest.mark.asyncio
async def test_request_limit_prompt_continues_with_native_usage_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
request_limit = 1
request_limit_prompt = "Check your progress and continue."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    agent = LLMBaseAgent(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )
    tool_executions = 0

    def ping():
        nonlocal tool_executions
        tool_executions += 1
        return "pong"

    agent.add_local_tool(ping)
    requests = []

    async def stream_function(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            yield {0: DeltaToolCall(name="ping", json_args="{}")}
        else:
            yield "done"

    async with _managed_runtime(agent, config_loader, agent.io_adapter):
        with agent.pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            result = await agent.step("start")

    assert result.output == "done"
    assert agent.session.result == "done"
    assert agent.session.error is None
    assert len(requests) == 2
    assert tool_executions == 1
    parts = [part for message in result.all_messages() for part in message.parts]
    assert any(isinstance(part, ToolCallPart) for part in parts)
    assert any(
        isinstance(part, ToolReturnPart) and part.content == "pong" for part in parts
    )
    user_prompts = [part.content for part in parts if isinstance(part, UserPromptPart)]
    assert not isinstance(user_prompts[0], str)
    assert isinstance(user_prompts[0][0], TextContent)
    assert user_prompts[0][0].content == "start"
    assert user_prompts[1] == "Check your progress and continue."


@pytest.mark.asyncio
async def test_agent_lifecycle_session_binding_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-id"], agent_name="test_agent")

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    assert agent.uuid == session.id == "test-session-id"
    assert session.runner is None
    assert session.is_active is False
    assert not hasattr(agent, "status")
    assert agent.uuid not in io_adapter.hosts

    async with _managed_runtime(agent, config_loader, io_adapter):
        assert session.agent is agent
        assert session.is_active is True
        assert agent.uuid in io_adapter.hosts

    assert session.runner is None
    assert session.is_active is False
    assert agent.uuid not in io_adapter.hosts


@pytest.mark.asyncio
async def test_agent_manages_current_task(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    agent = LLMBaseAgent(
        app_setup(cli_args={"workspace": str(tmp_path)}),
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["managed-task"], agent_name="test_agent"),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_step(user_input=None):
        started.set()
        await release.wait()
        return user_input

    agent.step = blocking_step  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    runner = TaskRunner(agent.session, agent.config_loader, agent.io_adapter)
    runner.runtime = agent
    agent.session.runner = runner
    async with agent:
        task = runner.run("work")
        await started.wait()
        assert runner.current_task is task
        with pytest.raises(RuntimeError, match="already running"):
            runner.run("duplicate")
        with pytest.raises(TimeoutError):
            await runner.wait(0.01)
        assert runner.current_task is task
        assert await runner.cancel()
        assert task.cancelled()
        assert runner.current_task is None
        assert not await runner.cancel()
    agent.session.runner = None


@pytest.mark.asyncio
async def test_agent_lifecycle_exception_sets_error_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-err"], agent_name="test_agent")

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    with pytest.raises(RuntimeError, match="something broke"):
        async with _managed_runtime(agent, config_loader, io_adapter):
            raise RuntimeError("something broke")

    assert session.runner is None
    assert session.events[-1].event_type == "error"
    assert "RuntimeError: something broke" in session.events[-1].error
    assert agent.uuid not in io_adapter.hosts


@pytest.mark.asyncio
async def test_agent_lifecycle_cancellation_sets_interrupted_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-cancel"], agent_name="test_agent")

    agent = LLMBaseAgent(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    with pytest.raises(asyncio.CancelledError):
        async with _managed_runtime(agent, config_loader, io_adapter):
            raise asyncio.CancelledError()

    assert session.runner is None
    assert session.events[-1].event_type == "error"
    assert session.events[-1].error == "Task interrupted."
    assert agent.uuid not in io_adapter.hosts


@pytest.mark.asyncio
async def test_runner_creates_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(
        path=["parent-id", "child-session-id"],
        agent_name="test_agent",
    )

    runner = TaskRunner(session, config_loader, io_adapter)
    agent = await runner.start()
    try:
        assert agent.uuid == "child-session-id"
        assert agent.session is session
        assert agent.name == "test_agent"
    finally:
        await runner.stop()
