import pytest

from arox.core.config import load_config, parse_dot_config


def test_config_basic_parsing(tmp_path):
    """Test basic config file parsing"""
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    model_ref = "test-model"
    [agent.test_agent]
    type = "chat"
    """)

    config = load_config([config_file])

    assert config.model_ref == "test-model"
    assert config.agent["test_agent"].type == "chat"


def test_config_override_order(tmp_path):
    """Test config file precedence"""
    file1 = tmp_path / "f1.toml"
    file1.write_text("model_ref = 'first'")

    file2 = tmp_path / "f2.toml"
    file2.write_text("model_ref = 'second'")

    config = load_config([file1, file2])
    assert config.model_ref == "second"  # Last file should win


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
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    model_ref = "file-model"
    [agent.test_agent]
    type = "chat"
    """)

    cli_overrides = parse_dot_config(
        ["model_ref=cli-model", "agent.test_agent.type=custom"]
    )
    config = load_config([config_file], cli_args=cli_overrides)

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

    host = tmp_path / "app.toml"
    host.write_text("""
    include = ["shared.toml"]
    [agent.compaction]
    system_prompt = "host-override"
    """)

    config = load_config([host])
    assert config.model_ref == "shared-model"
    comp = config.agent["compaction"]
    assert comp.type == "compaction"
    assert comp.system_prompt == "host-override"
    assert comp.task_prompt == "shared-task"


def test_config_include_circular_raises(tmp_path):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text('include = ["b.toml"]\n')
    b.write_text('include = ["a.toml"]\n')

    with pytest.raises(ValueError, match="Circular config include"):
        load_config([a])


def test_config_include_missing_raises(tmp_path):
    host = tmp_path / "app.toml"
    host.write_text('include = ["does_not_exist.toml"]\n')
    with pytest.raises(FileNotFoundError):
        load_config([host])


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
    (workspace / ".arox.config.toml").write_text("model_ref = 'workspace'")

    # 3. Explicit config
    explicit_file = explicit / "config.toml"
    explicit_file.write_text("model_ref = 'explicit'")

    # Load with only XDG
    config1 = load_config(workspace=workspace)
    assert config1.model_ref == "workspace"

    # Load without workspace config
    (workspace / ".arox.config.toml").unlink()
    config2 = load_config(workspace=workspace)
    assert config2.model_ref == "xdg"

    # Load with explicit
    (workspace / ".arox.config.toml").write_text("model_ref = 'workspace'")
    config3 = load_config(config_files=[explicit_file], workspace=workspace)
    assert config3.model_ref == "explicit"


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
    config1 = load_config(workspace=workspace)
    assert config1.agent["myagent"].system_prompt == "global-config"

    # Test 2: Global Config < Workspace Agents
    (workspace / ".agents").mkdir()
    (workspace / ".agents" / "myagent.md").write_text(
        "---\nsystem_prompt: workspace-agent\n---\n"
    )
    config2 = load_config(workspace=workspace)
    assert config2.agent["myagent"].system_prompt == "workspace-agent"

    # Test 3: Workspace Agents < Workspace Config
    (workspace / ".arox.config.toml").write_text(
        '[agent.myagent]\nsystem_prompt = "workspace-config"'
    )
    config3 = load_config(workspace=workspace)
    assert config3.agent["myagent"].system_prompt == "workspace-config"
