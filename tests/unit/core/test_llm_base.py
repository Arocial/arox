from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from arox.core.app import app_setup
from arox.core.io import AbstractIOAdapter, RequestEvent
from arox.core.llm_base import LLMBaseAgent, _complete_pending_tool_calls
from arox.core.session import AgentSession
from arox.plugins.core import SetModelEvent


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_io, event):
        pass


def test_complete_pending_tool_calls_noop_when_all_matched():
    messages = [
        ModelRequest(parts=[UserPromptPart("hi")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="t", args="{}", tool_call_id="c1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="t", content="ok", tool_call_id="c1")]
        ),
        ModelResponse(parts=[TextPart("done")]),
    ]
    before = len(messages)
    _complete_pending_tool_calls(messages)
    assert len(messages) == before


def test_complete_pending_tool_calls_fills_orphans():
    messages = [
        ModelRequest(parts=[UserPromptPart("hi")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="t1", args="{}", tool_call_id="c1"),
                ToolCallPart(tool_name="t2", args="{}", tool_call_id="c2"),
            ]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="t1", content="ok", tool_call_id="c1")]
        ),
    ]
    _complete_pending_tool_calls(messages)
    assert isinstance(messages[-1], ModelRequest)
    returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].tool_call_id == "c2"
    assert returns[0].tool_name == "t2"


@pytest.mark.asyncio
async def test_agent_skills_filtering(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".arox" / "skills"
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
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = ["skill1"]
""")

    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent(
        parsed_config,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
            agent_config=parsed_config.agent["test_agent"],
        ),
    )

    async with agent:
        assert "skill1" in agent.system_prompt
    assert "skill2" not in agent.system_prompt


@pytest.mark.asyncio
async def test_agent_skills_string(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".arox" / "skills"
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
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = "skill2"
""")

    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent(
        parsed_config,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
            agent_config=parsed_config.agent["test_agent"],
        ),
    )

    async with agent:
        assert "skill1" not in agent.system_prompt
    assert "skill2" in agent.system_prompt


@pytest.mark.asyncio
async def test_agent_skills_none(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".arox" / "skills"
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
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
""")

    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent(
        parsed_config,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
            agent_config=parsed_config.agent["test_agent"],
        ),
    )

    async with agent:
        assert "skill1" in agent.system_prompt
    assert "skill2" in agent.system_prompt


@pytest.mark.asyncio
async def test_request_event_dispatches_to_handler(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    class CustomEvent(RequestEvent):
        pass

    received: list[RequestEvent] = []

    agent = LLMBaseAgent(
        parsed_config,
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
async def test_set_model_event_updates_model_ref(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    agent = LLMBaseAgent(
        parsed_config,
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
