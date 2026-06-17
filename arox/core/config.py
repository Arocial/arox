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
    skills: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, ModelConfig] = Field(default_factory=dict)
    provider: dict[str, ProviderConfig] = Field(default_factory=dict)


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
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


def _discover_agents_in_scopes(scopes: list[Path]) -> dict[str, dict[str, Any]]:
    """Discover agent configurations from `.agents` directories in given scopes."""
    agent_configs: dict[str, Any] = {}

    from arox.utils.markdown import parse_yaml_frontmatter

    for scope in scopes:
        if not scope.exists() or not scope.is_dir():
            continue

        for file_path in scope.iterdir():
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            agent_name = file_path.stem

            if suffix == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                metadata, body = parse_yaml_frontmatter(content)
                metadata = metadata or {}

                if body and "system_prompt" not in metadata:
                    metadata["system_prompt"] = body

                if metadata:
                    agent_configs[agent_name] = deep_merge(
                        agent_configs.get(agent_name, {}), metadata
                    )

    return {"agent": agent_configs} if agent_configs else {}


def _discover_skills_in_scopes(scopes: list[Path]) -> dict[str, dict[str, Any]]:
    """Discover skills from SKILL.md files in given scopes."""
    skills: dict[str, Any] = {}

    import logging

    from arox.utils.markdown import parse_yaml_frontmatter

    logger = logging.getLogger(__name__)

    for scope in scopes:
        if not scope.exists() or not scope.is_dir():
            continue

        for skill_dir in scope.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists() or not skill_file.is_file():
                continue

            try:
                content = skill_file.read_text(encoding="utf-8")
                metadata, _ = parse_yaml_frontmatter(content)

                if (
                    not isinstance(metadata, dict)
                    or "name" not in metadata
                    or "description" not in metadata
                ):
                    logger.warning(f"Missing required metadata in {skill_file}")
                    continue

                name = metadata["name"]
                if name not in skills:
                    skills[name] = {
                        "name": name,
                        "description": metadata["description"],
                        "location": str(skill_file.absolute()),
                    }
            except Exception as e:
                logger.warning(f"Error reading skill file {skill_file}: {e}")

    return {"skills": skills} if skills else {}


def load_config(
    app_name: str | None = None,
    profile: str | Path | None = None,
    cli_args: list[str] | dict[str, Any] | None = None,
    workspace: Path | None = None,
) -> Config:
    import os

    xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    workspace = workspace if workspace else Path.cwd()

    raw_config: dict[str, Any] = {}

    # --- 1. Global Scope ---
    global_agent_scopes = [
        Path.home() / ".arox" / "agents",
        xdg_config_home / "arox" / "agents",
        Path.home() / ".agents",
    ]
    raw_config = deep_merge(raw_config, _discover_agents_in_scopes(global_agent_scopes))

    global_skill_scopes = [
        Path.home() / ".arox" / "skills",
        xdg_config_home / "arox" / "skills",
        Path.home() / ".agents" / "skills",
    ]
    raw_config = deep_merge(raw_config, _discover_skills_in_scopes(global_skill_scopes))

    for path in _discover_config_files(xdg_config_home / "arox", "config"):
        if path.exists():
            raw_config = deep_merge(raw_config, _load_config_file(path))

    # --- 2. App Scope ---
    if app_name and profile:
        profile_path = Path(profile)
        if profile_path.is_absolute():
            p_dir = profile_path
        else:
            user_profile_dir = (
                xdg_config_home / "arox" / "profiles" / app_name / profile_path
            )
            p_dir = user_profile_dir

        app_agent_scopes = [p_dir / ".agents", p_dir / "agents"]
        raw_config = deep_merge(
            raw_config, _discover_agents_in_scopes(app_agent_scopes)
        )

        app_skill_scopes = [p_dir / ".skills", p_dir / "skills"]
        raw_config = deep_merge(
            raw_config, _discover_skills_in_scopes(app_skill_scopes)
        )

        for path in _discover_config_files(p_dir, "config"):
            if path.exists():
                raw_config = deep_merge(raw_config, _load_config_file(path))

    # --- 3. Workspace Scope ---
    workspace_agent_scopes = [
        workspace / ".arox" / "agents",
        workspace / ".agents",
    ]
    raw_config = deep_merge(
        raw_config, _discover_agents_in_scopes(workspace_agent_scopes)
    )

    workspace_skill_scopes = [
        workspace / ".arox" / "skills",
        workspace / ".agents" / "skills",
    ]
    raw_config = deep_merge(
        raw_config, _discover_skills_in_scopes(workspace_skill_scopes)
    )

    for path in _discover_config_files(workspace, ".arox.config"):
        if path.exists():
            raw_config = deep_merge(raw_config, _load_config_file(path))

    # --- 4. CLI Overrides ---
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
