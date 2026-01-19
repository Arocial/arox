from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, TextIO

from arox.utils import deep_merge


def parse_dot_config(cli_args: list[str]) -> dict[str, Any]:
    """Parse arbitrary configs in dot notation to a nested dictionary.

    For example: ["a.b=value", "a.e.f=True"] will be parsed to:
    {
        "a": {
            "b": "value",
            "e": {
                "f": True
            }
        }
    }

    Args:
        cli_args: List of strings in the format "key.path=value".

    Returns:
        dict: Nested dictionary representing the parsed config.
    """
    result: dict[str, Any] = {}
    for arg in cli_args:
        if "=" not in arg:
            continue
        key_path, value = arg.split("=", 1)
        keys = [k.strip() for k in key_path.split(".")]
        if not keys or not keys[0]:
            continue

        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Convert value to appropriate type
        val_lower = value.lower()
        if val_lower == "true":
            parsed_value: Any = True
        elif val_lower == "false":
            parsed_value = False
        elif val_lower in ("none", "null"):
            parsed_value = None
        else:
            try:
                if "." in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                parsed_value = value
        current[keys[-1]] = parsed_value
    return result


class TomlConfigParser:
    def __init__(
        self,
        config_files: list[str | Path] | None = None,
        override_configs: dict[str, Any] | None = None,
    ):
        self._raw_data: dict[str, Any] = {}
        self.known_groups: list[ArgumentGroup] = []
        self.parsed = Config()
        self.default_group_name = "DEFAULT"
        self.default_group = self.add_argument_group(self.default_group_name)

        self.config_files = [Path(f) for f in config_files] if config_files else []
        self.override_configs = override_configs
        self.config_dirs: list[Path] = []

    def find_config(self, fpath: str | Path) -> Path | None:
        fpath = Path(fpath)
        if fpath.is_absolute():
            return fpath if fpath.exists() else None

        # Check current directory first
        if fpath.exists():
            return fpath.absolute()

        for directory in self.config_dirs:
            full_path = directory / fpath
            if full_path.exists():
                return full_path.absolute()
        return None

    def parse_args(self) -> Config:
        self.load_config()
        for group in self.known_groups:
            group.parse_args()

        # Move default group items to root
        if self.default_group_name in self.parsed:
            default_items = self.parsed.pop(self.default_group_name)
            self.parsed.update(default_items)

        self._validate_required()
        return self.parsed

    def _validate_required(self) -> None:
        for group in self.known_groups:
            for name, info in group.known_args.items():
                if info["required"]:
                    # Check in the group's own parsed Config object
                    if name not in group.parsed or group.parsed[name] is None:
                        group_prefix = (
                            f"[{group.name}] "
                            if group.name != self.default_group_name
                            else ""
                        )
                        raise ValueError(
                            f"Missing required configuration: {group_prefix}{name}"
                        )

    def add_argument_group(
        self, name: str, help: str = "", expose_raw: bool = False
    ) -> ArgumentGroup:
        """Create an argument group for organizing related arguments in TOML tables"""
        group = ArgumentGroup(self, name, help, expose_raw)
        self.known_groups.append(group)
        return group

    def add_argument(
        self, name: str, default: Any = None, help: str = "", required: bool = False
    ) -> None:
        """Add a known argument with optional default value"""
        self.default_group.add_argument(name, default, help, required)

    def dump_default_config(self, dest: TextIO | None = None) -> str:
        """Generate a default config file based on known arguments"""
        config = "\n".join([group.dump_default_config() for group in self.known_groups])
        if dest:
            dest.write(config)
        return config

    def load_config(self) -> dict[str, Any]:
        """Find and load TOML config file from various locations."""
        search_paths: list[Path] = []
        if self.config_files:
            search_paths.extend(self.config_files)

        # Standard locations
        search_paths.append(Path.home() / ".config" / "arox" / "config.toml")
        search_paths.append(Path.cwd() / ".arox.config.toml")

        # Update config_dirs for find_config
        self.config_dirs = list(dict.fromkeys([f.parent for f in search_paths]))

        config: dict[str, Any] = {}
        for path in search_paths:
            if path.exists():
                with open(path, "rb") as f:
                    config = deep_merge(config, tomllib.load(f))

        if self.override_configs:
            config = deep_merge(config, self.override_configs)

        self._raw_data = config
        return config


class ArgumentGroup:
    """Helper class for grouping arguments in TOML tables"""

    def __init__(
        self,
        parent: TomlConfigParser,
        name: str,
        help: str = "",
        expose_raw: bool = False,
    ):
        self.parent = parent
        self.name = name
        self.path = self._split_name(name)
        self.known_args: dict[str, dict[str, Any]] = {}
        self.help = help
        self.parsed = Config()
        self._raw_data: dict[str, Any] | None = None
        self.expose_raw = expose_raw

    def _split_name(self, name: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        in_quotes = False
        quote_char = None
        for char in name:
            if char in ('"', "'"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current.append(char)
            elif char == "." and not in_quotes:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return [p for p in parts if p]

    def parse_args(self) -> Config:
        self._parse_group()
        for name, info in self.known_args.items():
            self._parse_argument(name, info["default"])
        return self.parsed

    def _parse_group(self) -> None:
        raw: Any = self.parent._raw_data
        for part in self.path:
            if isinstance(raw, dict) and part in raw:
                raw = raw[part]
            else:
                raw = {}
                break
        self._raw_data = raw

        parsed = self.parent.parsed
        for part in self.path:
            parsed = parsed.setdefault(part, Config())

        if self.expose_raw and isinstance(self._raw_data, dict):
            parsed.update(self._raw_data)
        self.parsed = parsed

    def _parse_argument(self, name: str, default: Any) -> None:
        value = default
        if isinstance(self._raw_data, dict) and name in self._raw_data:
            value = self._raw_data[name]
        self.parsed[name] = value

    def add_argument(
        self, name: str, default: Any = None, help: str = "", required: bool = False
    ) -> None:
        """Add an argument to this group"""
        self.known_args[name] = {
            "default": default,
            "help": help,
            "required": required,
        }

    def dump_default_config(self) -> str:
        """Generate a default config file based on known arguments"""
        import json

        config_text = f"[{self.name}]\n"
        if self.help:
            config_text = f"# {self.help}\n" + config_text

        for name, info in self.known_args.items():
            help_text = info["help"].replace("\n", "\n# ")
            if help_text:
                config_text += f"# {help_text}\n"
            if info["required"]:
                config_text += "# Required: Yes\n"

            default = info["default"]
            if default is None:
                config_text += f"# {name} = \n\n"
            else:
                formatted_val = json.dumps(default)
                config_text += f"# {name} = {formatted_val}\n\n"

        return config_text


class Config(dict[str, Any]):
    """Wrapper class that allows both dot notation and dictionary-style access to fields.
    Nested dictionaries are automatically converted to Config objects.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = value

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, other: Any = None, **kwargs: Any) -> None:
        if other is not None:
            if hasattr(other, "items"):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def to_dict(self) -> dict[str, Any]:
        """Convert Config and nested Configs back to plain dictionaries."""
        result: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({super().__repr__()})"
