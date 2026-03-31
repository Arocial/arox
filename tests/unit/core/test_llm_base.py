from pathlib import Path

import pytest

from arox.core.app import app_setup
from arox.core.llm_base import LLMBaseAgent
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
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = ["skill1"]
""")

    parsed_config = app_setup(
        config_files=[config_file],
        cli_args={"workspace": str(tmp_path)},
    )

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", parsed_config, agent_io=io_channel)

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

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", parsed_config, agent_io=io_channel)

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

    io_channel = IOChannel()

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    agent = LLMBaseAgent("test_agent", parsed_config, agent_io=io_channel)

    assert "skill1" in agent.system_prompt
    assert "skill2" in agent.system_prompt
