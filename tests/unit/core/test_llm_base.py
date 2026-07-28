from pathlib import Path

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
from arox.core.session import AgentSession
from arox.plugins.core import SetModelEvent


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_io, event):
        pass


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

    async with agent:
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

    async with agent:
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

    async with agent:
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

    async with agent:
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

    async with agent:
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

    async with agent:
        with agent.pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            result = await agent.step("start")

    assert result.output == "done"
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
