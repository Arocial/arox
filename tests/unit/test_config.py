import pytest

from arox.config import ArgumentGroup, Config, TomlConfigParser


def test_config_basic_parsing(tmp_path):
    """Test basic config file parsing"""
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    [DEFAULT]
    foo = "bar"
    nested = { key = "value" }
    """)

    parser = TomlConfigParser([config_file])
    parser.add_argument("foo")
    parser.add_argument("nested")
    config = parser.parse_args()

    assert config.foo == "bar"
    assert config.nested.key == "value"


def test_config_group_parsing(tmp_path):
    """Test parsing with nested groups"""
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    [group.subgroup]
    value = 42
    """)

    parser = TomlConfigParser([config_file])
    group = parser.add_argument_group("group.subgroup")
    group.add_argument("value", default=0)

    config = parser.parse_args()
    assert config.group.subgroup.value == 42


def test_config_default_values():
    """Test default values when config is missing"""
    parser = TomlConfigParser()
    parser.add_argument("missing", default="default_value")
    parser.add_argument_group("group").add_argument("nested", default=123)

    config = parser.parse_args()
    assert config.missing == "default_value"
    assert config.group.nested == 123


def test_config_override_order(tmp_path):
    """Test config file precedence"""
    file1 = tmp_path / "f1.toml"
    file1.write_text("[DEFAULT]\nvalue = 'first'")

    file2 = tmp_path / "f2.toml"
    file2.write_text("[DEFAULT]\nvalue = 'second'")

    parser = TomlConfigParser([file1, file2])
    parser.add_argument("value", default="default")

    config = parser.parse_args()
    assert config.value == "second"  # Last file should win


def test_config_dict_access():
    """Test Config class dict-style access"""
    config = Config({"key": "value", "nested": {"inner": 42}})

    assert config["key"] == "value"
    assert config["nested"]["inner"] == 42
    assert config.nested.inner == 42


def test_config_missing_attribute():
    """Test Config class missing attribute handling"""
    config = Config({})

    with pytest.raises(AttributeError):
        _ = config.missing


def test_argument_group_dump_config():
    """Test generating default config from argument group"""
    parser = TomlConfigParser()
    group = ArgumentGroup(parser, "test")
    group.add_argument("param1", default=1, help="First param")
    group.add_argument("param2", default="two", help="Second param")

    config = group.dump_default_config()
    assert "[test]" in config
    assert "# param1 = 1" in config
    assert '# param2 = "two"' in config


def test_expose_raw_config(tmp_path):
    """Test expose_raw flag for argument groups"""
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    [group]
    value = 42
    extra = "should_be_passed_through"
    """)

    parser = TomlConfigParser([config_file])
    group = parser.add_argument_group("group", expose_raw=True)
    group.add_argument("value", default=0)

    config = parser.parse_args()
    assert config.group.value == 42
    assert config.group.extra == "should_be_passed_through"


def test_parse_dot_config():
    """Test parse_nested_config function"""
    from arox.config import parse_dot_config

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
