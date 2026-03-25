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
