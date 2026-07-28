from pathlib import Path

import pytest

from arox.core.config import ConfigLoader, ProviderConfig, parse_dot_config


def test_provider_header_defaults():
    provider = ProviderConfig()

    assert provider.session_header == "X-Session-Id"
    assert provider.turn_header == "X-Turn-Id"
    assert provider.disabled_native_tools == []


def test_config_basic_parsing(tmp_path):
    """Test basic config file parsing"""
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
    model_ref = "test-model"
    [agent.test_agent]
    type = "chat"
    """)

    config = ConfigLoader(workspace=tmp_path).reload()

    assert config.model_ref == "test-model"
    assert config.agent["test_agent"].type == "chat"


def test_config_override_order(tmp_path):
    """Test config file precedence using workspace"""
    ws1 = tmp_path / "ws1"
    ws1.mkdir()
    ws1_arox = ws1 / ".arox"
    ws1_arox.mkdir(exist_ok=True)
    (ws1_arox / "config.toml").write_text("model_ref = 'second'")

    config = ConfigLoader(workspace=ws1).reload()
    assert config.model_ref == "second"


def test_parse_dot_config():
    """Test parse_nested_config function"""
    # Test basic nested structure
    args = ["a.b=value", "a.e.f=True", "a.e.g=42", "a.e.h=3.14"]
    result = parse_dot_config(args)
    assert result == {"a": {"b": "value", "e": {"f": True, "g": 42, "h": 3.14}}}

    # Test type conversion
    args = ["bool.true=true", "bool.false=false", "number.int=123", "number.float=1.23"]
    result = parse_dot_config(args)
    assert result == {
        "bool": {"true": True, "false": False},
        "number": {"int": 123, "float": 1.23},
    }

    # Test malformed entries
    args = ["valid.key=value", "invalid_entry", "another.valid=123"]
    result = parse_dot_config(args)
    assert result == {"valid": {"key": "value"}, "another": {"valid": 123}}


def test_cli_overrides(tmp_path):
    """Test CLI overrides merging with file config"""
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
    model_ref = "file-model"
    [agent.test_agent]
    type = "chat"
    """)

    cli_overrides = parse_dot_config(
        ["model_ref=cli-model", "agent.test_agent.type=custom"]
    )
    config = ConfigLoader(workspace=tmp_path, cli_args=cli_overrides).reload()

    assert config.model_ref == "cli-model"
    assert config.agent["test_agent"].type == "custom"


def test_config_include_merges_and_overrides(tmp_path):
    """Top-level `include` merges referenced files; host overrides them."""
    shared = tmp_path / "shared.toml"
    shared.write_text("""
    model_ref = "shared-model"
    [agent.compaction]
    type = "compaction"
    system_prompt = "shared"
    task_prompt = "shared-task"
    """)

    host = tmp_path / ".arox" / "config.toml"
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("""
    include = ["../shared.toml"]
    [agent.compaction]
    system_prompt = "host-override"
    """)

    config = ConfigLoader(workspace=tmp_path).reload()
    assert config.model_ref == "shared-model"
    comp = config.agent["compaction"]
    assert comp.type == "compaction"
    assert comp.system_prompt == "host-override"
    assert comp.task_prompt == "shared-task"


def test_config_include_circular_raises(tmp_path):
    a = tmp_path / ".arox" / "config.toml"
    a.parent.mkdir(parents=True, exist_ok=True)
    b = tmp_path / "b.toml"
    a.write_text('include = ["../b.toml"]\n')
    b.write_text('include = [".arox/config.toml"]\n')

    with pytest.raises(ValueError, match="Circular config include"):
        ConfigLoader(workspace=tmp_path).reload()


def test_config_include_missing_raises(tmp_path):
    host = tmp_path / ".arox" / "config.toml"
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text('include = ["does_not_exist.toml"]\n')
    with pytest.raises(FileNotFoundError):
        ConfigLoader(workspace=tmp_path).reload()


def test_config_search_paths_precedence(tmp_path, monkeypatch):
    """Test precedence: XDG_CONFIG_HOME < workspace < config_files < cli_args"""
    xdg_config = tmp_path / "xdg"
    workspace = tmp_path / "ws"
    explicit = tmp_path / "explicit"

    xdg_config.mkdir()
    workspace.mkdir()
    explicit.mkdir()

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))

    # 1. XDG Config
    (xdg_config / "arox").mkdir()
    (xdg_config / "arox" / "config.toml").write_text("model_ref = 'xdg'")

    # 2. Workspace
    (workspace / ".arox").mkdir(exist_ok=True)
    (workspace / ".arox" / "config.toml").write_text("model_ref = 'workspace'")

    # 3. Explicit config
    explicit_file = explicit / "config.toml"
    explicit_file.write_text("model_ref = 'explicit'")

    # Load with only XDG
    config1 = ConfigLoader(workspace=workspace).reload()
    assert config1.model_ref == "workspace"

    # Load without workspace config
    (workspace / ".arox" / "config.toml").unlink()
    config2 = ConfigLoader(workspace=workspace).reload()
    assert config2.model_ref == "xdg"

    # Load with CLI args
    (workspace / ".arox" / "config.toml").write_text("model_ref = 'workspace'")
    config3 = ConfigLoader(cli_args=["model_ref=cli"], workspace=workspace).reload()
    assert config3.model_ref == "cli"


def test_config_interleaved_scope_precedence(tmp_path, monkeypatch):
    """Test interleaved scope precedence:
    Global Agents < Global Config < Workspace Agents < Workspace Config
    """
    xdg_config = tmp_path / "xdg"
    workspace = tmp_path / "ws"
    xdg_config.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))

    # Global Setup
    (xdg_config / "arox").mkdir()
    (xdg_config / "arox" / "agents").mkdir()
    (xdg_config / "arox" / "agents" / "myagent.md").write_text(
        "---\nsystem_prompt: global-agent\n---\n"
    )
    (xdg_config / "arox" / "config.toml").write_text(
        '[agent.myagent]\nsystem_prompt = "global-config"'
    )

    # Test 1: Global Agents < Global Config
    config1 = ConfigLoader(workspace=workspace).reload()
    assert config1.agent["myagent"].system_prompt == "global-config"

    # Test 2: Global Config < Workspace Agents
    (workspace / ".agents").mkdir()
    (workspace / ".agents" / "agents").mkdir()
    (workspace / ".agents" / "agents" / "myagent.md").write_text(
        "---\nsystem_prompt: workspace-agent\n---\n"
    )
    config2 = ConfigLoader(workspace=workspace).reload()
    assert config2.agent["myagent"].system_prompt == "workspace-agent"

    # Test 3: Workspace Agents < Workspace Config
    (workspace / ".arox").mkdir(exist_ok=True)
    (workspace / ".arox" / "config.toml").write_text(
        '[agent.myagent]\nsystem_prompt = "workspace-config"'
    )
    config3 = ConfigLoader(workspace=workspace).reload()
    assert config3.agent["myagent"].system_prompt == "workspace-config"


def test_discover_skills_empty(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home_dir / ".config"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = ConfigLoader(workspace=workspace).reload()
    assert config.skills == {}


def test_discover_skills_valid(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home_dir / ".config"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".agents" / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: test_skill\ndescription: A test skill\n---\nSkill content here",
        encoding="utf-8",
    )

    config = ConfigLoader(workspace=workspace).reload()
    assert "test_skill" in config.skills
    assert config.skills["test_skill"]["name"] == "test_skill"
    assert config.skills["test_skill"]["description"] == "A test skill"
    assert config.skills["test_skill"]["location"] == str(skill_file.absolute())


def test_discover_skills_malformed_yaml_fixed(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home_dir / ".config"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".agents" / "skills" / "malformed_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: malformed_skill\n"
        "description: A skill with a colon: in description\n"
        "---\n"
        "Content",
        encoding="utf-8",
    )

    config = ConfigLoader(workspace=workspace).reload()
    assert "malformed_skill" in config.skills
    assert (
        config.skills["malformed_skill"]["description"]
        == "A skill with a colon: in description"
    )


def test_discover_skills_missing_metadata(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home_dir / ".config"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_dir = workspace / ".agents" / "skills" / "missing_meta"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: missing_meta\n---\nContent", encoding="utf-8")

    config = ConfigLoader(workspace=workspace).reload()
    assert "missing_meta" not in config.skills


def test_loader_returns_cached_snapshot_when_sources_are_unchanged(tmp_path):
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "cached"\n')
    loader = ConfigLoader(workspace=tmp_path)

    first = loader.reload()
    second = loader.reload()
    forced = loader.reload(force=True)

    assert second is first
    assert forced is not first
    assert forced == first


def test_loader_reloads_modified_config(tmp_path):
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "before"\n')
    loader = ConfigLoader(workspace=tmp_path)
    before = loader.reload()

    config_file.write_text('model_ref = "after-change"\n')
    result = loader.reload()

    assert result is loader.current_config
    assert result.model_ref == "after-change"
    assert result is not before


def test_loader_tracks_added_and_removed_agent_files(tmp_path):
    agent_dir = tmp_path / ".agents" / "agents"
    agent_dir.mkdir(parents=True)
    loader = ConfigLoader(workspace=tmp_path)
    initial = loader.reload()
    agent_file = agent_dir / "dynamic.md"

    agent_file.write_text("---\ntype: chat\n---\nDynamic prompt\n")
    added = loader.reload()

    assert added.agent["dynamic"].system_prompt == "Dynamic prompt"
    assert initial is not added

    agent_file.unlink()
    removed = loader.reload()

    assert "dynamic" not in removed.agent
    assert removed is not added


def test_loader_reloads_changed_include(tmp_path):
    shared = tmp_path / "shared.toml"
    shared.write_text('model_ref = "shared-before"\n')
    host = tmp_path / ".arox" / "config.toml"
    host.parent.mkdir(parents=True)
    host.write_text('include = ["../shared.toml"]\n')
    loader = ConfigLoader(workspace=tmp_path)
    loader.reload()

    shared.write_text('model_ref = "shared-after-change"\n')
    result = loader.reload()

    assert result.model_ref == "shared-after-change"


def test_loader_keeps_last_valid_snapshot_after_reload_error(tmp_path):
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "valid"\n')
    loader = ConfigLoader(workspace=tmp_path)
    valid = loader.reload()

    config_file.write_text('model_ref = ["invalid"\n')
    failed = loader.reload()

    assert failed is valid
    assert loader.last_error is not None
    assert loader.current_config is valid

    config_file.write_text('model_ref = "recovered-value"\n')
    recovered = loader.reload()

    assert loader.last_error is None
    assert recovered.model_ref == "recovered-value"
    assert recovered is not valid
