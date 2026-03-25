from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
            if key not in current or not isinstance(current[key], dict):
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


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enable: bool = False
    scrubbing: Literal[False] | None = None
    logfire: bool = False


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider_model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "chat"
    system_prompt: str = ""
    model_ref: str = ""
    plugins: list[str] = Field(default_factory=list)
    skills: str | list[str] | None = None
    examples: str | None = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    model_prompt: dict[str, str] = Field(default_factory=dict)
    pre_step_hooks: list[str] = Field(default_factory=list)
    post_step_hooks: list[str] = Field(default_factory=list)


class ComposerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    main_agent: str
    subagents: list[str] = Field(default_factory=list)
    io_adapter: str = "arox.ui.text_io.TextIOAdapter"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_ref: str = "deepseek/deepseek-chat"
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api_keys: dict[str, str] = Field(default_factory=dict)
    env_vars: dict[str, str] = Field(default_factory=dict)
    mcp_servers: dict[str, Any] = Field(default_factory=dict)

    composer: dict[str, ComposerConfig] = Field(default_factory=dict)
    agent: dict[str, AgentConfig] = Field(default_factory=dict)
    model: dict[str, ModelConfig] = Field(default_factory=dict)

    config_dirs: list[Path] = Field(default_factory=list, exclude=True)
    user_path: Path = Field(default_factory=lambda: Path.home() / ".arox", exclude=True)
    verbose_out_path: Path = Field(
        default_factory=lambda: Path.home() / ".arox" / "__verbose_out__", exclude=True
    )

    def find_file(self, fpath: str | Path) -> Path | None:
        fpath = Path(fpath)
        if fpath.is_absolute():
            return fpath if fpath.exists() else None

        if fpath.exists():
            return fpath.absolute()

        for directory in self.config_dirs:
            full_path = directory / fpath
            if full_path.exists():
                return full_path.absolute()
        return None


def load_config(
    config_files: list[str | Path] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    search_paths: list[Path] = []
    if config_files:
        search_paths.extend([Path(f) for f in config_files])

    search_paths.append(Path.home() / ".config" / "arox" / "config.toml")
    search_paths.append(Path.cwd() / ".arox.config.toml")

    config_dirs = list(dict.fromkeys([f.parent for f in search_paths]))

    raw_config: dict[str, Any] = {}
    for path in search_paths:
        if path.exists():
            with open(path, "rb") as f:
                raw_config = deep_merge(raw_config, tomllib.load(f))

    if cli_overrides:
        raw_config = deep_merge(raw_config, cli_overrides)

    app_config = AppConfig(**raw_config)
    app_config.config_dirs = config_dirs
    return app_config
