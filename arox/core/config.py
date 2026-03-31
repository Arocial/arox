from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

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
        arg = arg.removeprefix("--")
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
    enable: bool = False
    scrubbing: Literal[False] | None = None
    logfire: bool = False


class ModelConfig(BaseModel):
    provider_model: str = ""
    base_url: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
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
    main_agent: str
    subagents: list[str] = Field(default_factory=list)
    io_adapter: str = "arox.ui.text_io.TextIOAdapter"


class AppConfig(BaseModel):
    env_vars: dict[str, str] = Field(default_factory=dict)
    api_keys: dict[str, str] = Field(default_factory=dict)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


class Config(BaseModel):
    model_ref: str = "deepseek:deepseek-chat"
    app: AppConfig = Field(default_factory=AppConfig)
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    composer: dict[str, ComposerConfig] = Field(default_factory=dict)
    agent: dict[str, AgentConfig] = Field(default_factory=dict)
    model: dict[str, ModelConfig] = Field(default_factory=dict)


def load_config(
    config_files: list[str | Path] | None = None,
    cli_args: list[str] | dict[str, Any] | None = None,
    workspace: Path | None = None,
) -> tuple[Config, list[Path]]:
    search_paths: list[Path] = []
    if config_files:
        search_paths.extend([Path(f) for f in config_files])

    search_paths.append(Path.home() / ".config" / "arox" / "config.toml")

    workspace = workspace if workspace else Path.cwd()
    search_paths.append(workspace / ".arox.config.toml")

    config_dirs = list(dict.fromkeys([f.parent for f in search_paths]))

    raw_config: dict[str, Any] = {}
    for path in search_paths:
        if path.exists():
            with open(path, "rb") as f:
                raw_config = deep_merge(raw_config, tomllib.load(f))

    if cli_args is not None:
        if isinstance(cli_args, list):
            cli_overrides = parse_dot_config(cli_args)
        else:
            cli_overrides = cli_args
    else:
        cli_overrides = {}

    if cli_overrides:
        raw_config = deep_merge(raw_config, cli_overrides)

    parsed_config = Config(**raw_config)
    return parsed_config, config_dirs
