from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.settings import ModelSettings

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


class ProviderConfig(BaseModel):
    base_url: str = ""
    session_header: str = ""


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_model: str = ""
    params: ModelSettings = Field(default_factory=ModelSettings)
    compaction_threshold: int | float | None = None


class AgentConfig(BaseModel):
    type: str = "chat"
    description: str = ""
    system_prompt: str = ""
    task_prompt: str = ""
    model_ref: str = ""
    fallback_model_ref: str | list[str] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)
    skills: str | list[str] | None = None
    mcp_servers: str | list[str] | None = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    model_prompt: dict[str, str] = Field(default_factory=dict)
    subagents: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    main_agent: str = "coder"
    env_vars: dict[str, str] = Field(default_factory=dict)
    api_keys: dict[str, str] = Field(default_factory=dict)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    session_max_age_days: int = 30


class Config(BaseModel):
    model_ref: str = "deepseek:deepseek-chat"
    fallback_model_ref: str | list[str] = Field(default_factory=list)
    available_models: list[str] = Field(default_factory=list)
    compaction_threshold: int | float = 0.7
    app: AppConfig = Field(default_factory=AppConfig)
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, AgentConfig] = Field(default_factory=dict)
    model: dict[str, ModelConfig] = Field(default_factory=dict)
    provider: dict[str, ProviderConfig] = Field(default_factory=dict)


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def _load_config_file(path: Path, _visited: set[Path] | None = None) -> dict[str, Any]:
    """Load a config file and recursively resolve any top-level ``include``.

    ``include`` may be a string or list of strings. Paths are resolved relative
    to the file containing the directive. Included files are merged first; the
    host file's keys override anything they define.
    """
    resolved = path.resolve()
    visited = _visited if _visited is not None else set()
    if resolved in visited:
        raise ValueError(f"Circular config include detected: {resolved}")
    visited = visited | {resolved}

    raw = _read_config_file(path)
    includes = raw.pop("include", None)
    if includes is None:
        return raw

    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(
            f"`include` in {path} must be a string or list of strings, "
            f"got {type(includes).__name__}"
        )

    base_dir = resolved.parent
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = base_dir / inc_path
        if not inc_path.exists():
            raise FileNotFoundError(
                f"Included config file not found: {inc_path} (from {path})"
            )
        merged = deep_merge(merged, _load_config_file(inc_path, visited))
    return deep_merge(merged, raw)


def _discover_config_files(base: Path, stem: str) -> list[Path]:
    """Discover config files with the given stem in toml/yaml formats."""
    found = []
    for suffix in (".toml", ".yaml", ".yml"):
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            found.append(candidate)
    return found


def load_config(
    config_files: list[str | Path] | None = None,
    cli_args: list[str] | dict[str, Any] | None = None,
    workspace: Path | None = None,
) -> Config:
    search_paths: list[Path] = []
    if config_files:
        search_paths.extend([Path(f) for f in config_files])

    search_paths.extend(
        _discover_config_files(Path.home() / ".config" / "arox", "config")
    )

    workspace = workspace if workspace else Path.cwd()
    search_paths.extend(_discover_config_files(workspace, ".arox.config"))

    raw_config: dict[str, Any] = {}
    for path in search_paths:
        if path.exists():
            raw_config = deep_merge(raw_config, _load_config_file(path))

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
    return parsed_config
