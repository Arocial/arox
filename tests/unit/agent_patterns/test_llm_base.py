from pathlib import Path

import pytest

from arox import agent_patterns
from arox.agent_patterns.llm_base import LLMBaseAgent
from arox.config import TomlConfigParser
from arox.ui.io import IOChannel


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
[DEFAULT]
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = ["skill1"]
[agent.test_agent.model_params]
""")

    toml_parser = TomlConfigParser(
        config_files=[config_file],
        override_configs={"workspace": str(tmp_path)},
    )
    agent_patterns.init(toml_parser)

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", toml_parser, agent_io=io_channel)

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
[DEFAULT]
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = "skill2"
[agent.test_agent.model_params]
""")

    toml_parser = TomlConfigParser(
        config_files=[config_file],
        override_configs={"workspace": str(tmp_path)},
    )
    agent_patterns.init(toml_parser)

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", toml_parser, agent_io=io_channel)

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
[DEFAULT]
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
[agent.test_agent.model_params]
""")

    toml_parser = TomlConfigParser(
        config_files=[config_file],
        override_configs={"workspace": str(tmp_path)},
    )
    agent_patterns.init(toml_parser)

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", toml_parser, agent_io=io_channel)

    assert "skill1" in agent.system_prompt
    assert "skill2" in agent.system_prompt
