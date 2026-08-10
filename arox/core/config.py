from __future__ import annotations

import copy
import logging
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.settings import ModelSettings

from arox.utils import deep_merge

logger = logging.getLogger(__name__)


def parse_dot_config(cli_args: list[str]) -> dict[str, Any]:
    """Parse arbitrary configs in dot notation to a nested dictionary."""
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
    session_header: str = "X-Session-Id"
    turn_header: str = "X-Turn-Id"
    disabled_native_tools: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_model: str = ""
    params: ModelSettings = Field(default_factory=ModelSettings)
    compaction_threshold: int | float | None = None


class AgentConfig(BaseModel):
    description: str = ""
    system_prompt: str = ""
    task_prompt: str = ""
    model_ref: str = ""
    fallback_model_ref: str | list[str] = Field(default_factory=list)
    request_limit: int | None = Field(default=50, gt=0)
    request_limit_prompt: str | None = None
    plugins: list[str] = Field(default_factory=list)
    plugin_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    skills: str | list[str] | None = None
    default_skills: str | list[str] | None = None
    mcp_servers: str | list[str] | None = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    model_prompt: dict[str, str] = Field(default_factory=dict)
    subagents: list[str] = Field(default_factory=list)
    max_parallel_subagents: int = Field(default=4, ge=1, le=32)


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


@dataclass(frozen=True)
class FileFingerprint:
    mtime_ns: int
    size: int
    inode: int


@dataclass
class CachedFile:
    fingerprint: FileFingerprint
    value: dict[str, Any]


class ConfigLoader:
    """Load layered configuration and cache unchanged source files."""

    def __init__(
        self,
        app_name: str | None = None,
        profile: str | Path | None = None,
        cli_args: list[str] | dict[str, Any] | None = None,
        workspace: Path | None = None,
    ) -> None:
        from platformdirs import user_config_dir

        self.app_name = app_name
        self.profile = profile
        self.workspace = (workspace or Path.cwd()).resolve()
        self.user_config_path = Path(user_config_dir("arox")).resolve()
        self.home = Path.home().resolve()
        if isinstance(cli_args, list):
            self.cli_overrides = parse_dot_config(cli_args)
        else:
            self.cli_overrides = copy.deepcopy(cli_args or {})

        self._file_cache: dict[Path, CachedFile] = {}
        self._include_paths: set[Path] = set()
        self._source_fingerprints: dict[Path, FileFingerprint] = {}
        self._current_config: Config | None = None
        self._last_error: Exception | None = None
        self._lock = threading.RLock()

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def for_workspace(self, workspace: Path | str | None = None) -> "ConfigLoader":
        """Create an independent loader with the same context for a workspace."""
        return ConfigLoader(
            app_name=self.app_name,
            profile=self.profile,
            cli_args=self.cli_overrides,
            workspace=Path(workspace) if workspace is not None else self.workspace,
        )

    @property
    def current_config(self) -> Config:
        return (
            self._current_config if self._current_config is not None else self.reload()
        )

    def reload(self, *, force: bool = False) -> Config:
        """Return the current config, reloading changed sources when needed.

        Once a valid snapshot exists, a failed reload keeps and returns that
        snapshot.
        """
        with self._lock:
            paths = self._discover_source_paths() | self._include_paths
            fingerprints = self._fingerprints(paths)
            changed = fingerprints != self._source_fingerprints
            if self._current_config is not None and not force and not changed:
                return self._current_config
            if force:
                self._file_cache.clear()

            try:
                config, include_paths = self._build_config()
            except Exception as exc:
                self._last_error = exc
                if self._current_config is None:
                    raise
                logger.warning(
                    "Failed to reload config; keeping last valid snapshot: %s", exc
                )
                return self._current_config

            self._include_paths = include_paths
            all_paths = self._discover_source_paths() | include_paths
            self._source_fingerprints = self._fingerprints(all_paths)
            self._current_config = config
            self._last_error = None
            self._prune_file_cache(all_paths)
            return config

    def invalidate(self, paths: set[Path] | None = None) -> None:
        """Invalidate all cached files or selected source paths."""
        with self._lock:
            if paths is None:
                self._file_cache.clear()
                self._source_fingerprints.clear()
                return
            for path in paths:
                resolved = path.resolve()
                self._file_cache.pop(resolved, None)
                self._source_fingerprints.pop(resolved, None)

    def _profile_dir(self) -> Path | None:
        if not self.app_name or not self.profile:
            return None
        profile_path = Path(self.profile)
        if profile_path.is_absolute():
            return profile_path.resolve()
        return (
            self.user_config_path / "profiles" / self.app_name / profile_path
        ).resolve()

    def _scope_groups(self) -> list[tuple[list[Path], list[Path], list[Path]]]:
        global_scopes = [self.user_config_path, self.home / ".agents"]
        groups = [
            (
                [scope / "agents" for scope in global_scopes],
                [scope / "skills" for scope in global_scopes],
                [self.user_config_path],
            )
        ]

        profile_dir = self._profile_dir()
        if profile_dir is not None:
            groups.append(
                (
                    [profile_dir / "agents"],
                    [profile_dir / "skills"],
                    [profile_dir],
                )
            )

        workspace_scopes = [self.workspace / ".arox", self.workspace / ".agents"]
        groups.append(
            (
                [scope / "agents" for scope in workspace_scopes],
                [scope / "skills" for scope in workspace_scopes],
                [self.workspace / ".arox"],
            )
        )
        return groups

    def _discover_source_paths(self) -> set[Path]:
        paths: set[Path] = set()
        for agent_scopes, skill_scopes, config_scopes in self._scope_groups():
            for scope in agent_scopes:
                if scope.is_dir():
                    paths.update(
                        path.resolve()
                        for path in scope.iterdir()
                        if path.is_file() and path.suffix.lower() == ".md"
                    )
            for scope in skill_scopes:
                if scope.is_dir():
                    paths.update(
                        skill_file.resolve()
                        for skill_dir in scope.iterdir()
                        if skill_dir.is_dir()
                        and (skill_file := skill_dir / "SKILL.md").is_file()
                    )
            for scope in config_scopes:
                for suffix in (".toml", ".yaml", ".yml"):
                    candidate = scope / f"config{suffix}"
                    if candidate.is_file():
                        paths.add(candidate.resolve())
        return paths

    @staticmethod
    def _fingerprint(path: Path) -> FileFingerprint | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return FileFingerprint(stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def _fingerprints(self, paths: set[Path]) -> dict[Path, FileFingerprint]:
        return {
            path: fingerprint
            for path in paths
            if (fingerprint := self._fingerprint(path)) is not None
        }

    def _read_config_file(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        fingerprint = self._fingerprint(resolved)
        if fingerprint is None:
            raise FileNotFoundError(f"Config file not found: {resolved}")
        cached = self._file_cache.get(resolved)
        if cached and cached.fingerprint == fingerprint:
            return copy.deepcopy(cached.value)

        suffix = resolved.suffix.lower()
        if suffix == ".toml":
            with resolved.open("rb") as file:
                value = tomllib.load(file)
        elif suffix in (".yaml", ".yml"):
            with resolved.open("r", encoding="utf-8") as file:
                value = yaml.safe_load(file) or {}
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
        if not isinstance(value, dict):
            raise ValueError(f"Config file must contain a mapping: {resolved}")

        self._file_cache[resolved] = CachedFile(fingerprint, copy.deepcopy(value))
        return value

    def _load_config_file(
        self,
        path: Path,
        visited: tuple[Path, ...],
        include_paths: set[Path],
    ) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in visited:
            chain = " -> ".join(str(item) for item in (*visited, resolved))
            raise ValueError(f"Circular config include detected: {chain}")

        raw = self._read_config_file(resolved)
        includes = raw.pop("include", None)
        if includes is None:
            return raw
        if isinstance(includes, str):
            includes = [includes]
        if not isinstance(includes, list) or not all(
            isinstance(item, str) for item in includes
        ):
            raise ValueError(
                f"`include` in {resolved} must be a string or list of strings"
            )

        merged: dict[str, Any] = {}
        for include in includes:
            include_path = Path(include)
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            include_path = include_path.resolve()
            if not include_path.is_file():
                raise FileNotFoundError(
                    f"Included config file not found: {include_path} (from {resolved})"
                )
            include_paths.add(include_path)
            merged = deep_merge(
                merged,
                self._load_config_file(
                    include_path, (*visited, resolved), include_paths
                ),
            )
        return deep_merge(merged, raw)

    def _discover_configs(
        self, scopes: list[Path], include_paths: set[Path]
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for scope in scopes:
            if not scope.is_dir():
                continue
            for suffix in (".toml", ".yaml", ".yml"):
                candidate = scope / f"config{suffix}"
                if candidate.is_file():
                    merged = deep_merge(
                        merged,
                        self._load_config_file(candidate, (), include_paths),
                    )
        return merged

    def _read_markdown_frontmatter(
        self, path: Path
    ) -> tuple[dict[str, Any] | None, str]:
        from arox.utils.markdown import parse_yaml_frontmatter

        resolved = path.resolve()
        fingerprint = self._fingerprint(resolved)
        if fingerprint is None:
            raise FileNotFoundError(resolved)
        cached = self._file_cache.get(resolved)
        if cached and cached.fingerprint == fingerprint:
            value = copy.deepcopy(cached.value)
            return value["metadata"], value["body"]

        metadata, body = parse_yaml_frontmatter(resolved.read_text(encoding="utf-8"))
        value = {"metadata": metadata, "body": body}
        self._file_cache[resolved] = CachedFile(fingerprint, copy.deepcopy(value))
        return metadata, body

    def _discover_agents(self, scopes: list[Path]) -> dict[str, Any]:
        agent_configs: dict[str, Any] = {}
        for scope in scopes:
            if not scope.is_dir():
                continue
            for file_path in sorted(scope.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() != ".md":
                    continue
                metadata, body = self._read_markdown_frontmatter(file_path)
                metadata = metadata or {}
                if body and "system_prompt" not in metadata:
                    metadata["system_prompt"] = body
                if metadata:
                    agent_configs[file_path.stem] = deep_merge(
                        agent_configs.get(file_path.stem, {}), metadata
                    )
        return {"agent": agent_configs} if agent_configs else {}

    def _discover_skills(self, scopes: list[Path]) -> dict[str, Any]:
        skills: dict[str, Any] = {}
        for scope in scopes:
            if not scope.is_dir():
                continue
            for skill_dir in sorted(scope.iterdir()):
                skill_file = skill_dir / "SKILL.md"
                if not skill_dir.is_dir() or not skill_file.is_file():
                    continue
                try:
                    metadata, _ = self._read_markdown_frontmatter(skill_file)
                    if (
                        not isinstance(metadata, dict)
                        or "name" not in metadata
                        or "description" not in metadata
                    ):
                        logger.warning("Missing required metadata in %s", skill_file)
                        continue
                    name = metadata["name"]
                    if name not in skills:
                        skills[name] = {
                            "name": name,
                            "description": metadata["description"],
                            "location": str(skill_file.absolute()),
                        }
                except Exception as exc:
                    logger.warning("Error reading skill file %s: %s", skill_file, exc)
        return {"skills": skills} if skills else {}

    def _build_config(self) -> tuple[Config, set[Path]]:
        raw_config: dict[str, Any] = {}
        include_paths: set[Path] = set()
        for agent_scopes, skill_scopes, config_scopes in self._scope_groups():
            raw_config = deep_merge(raw_config, self._discover_agents(agent_scopes))
            raw_config = deep_merge(raw_config, self._discover_skills(skill_scopes))
            raw_config = deep_merge(
                raw_config, self._discover_configs(config_scopes, include_paths)
            )
        if self.cli_overrides:
            raw_config = deep_merge(raw_config, copy.deepcopy(self.cli_overrides))
        return Config(**raw_config), include_paths

    def _prune_file_cache(self, source_paths: set[Path]) -> None:
        stale = self._file_cache.keys() - source_paths
        for path in stale:
            del self._file_cache[path]
