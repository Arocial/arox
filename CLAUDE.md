#his file provides guidance to coding agents(e.g. Claude Code) when working with code in this repository.

## Development Commands

```bash
uv sync              # Install dependencies
uv run pytest        # Run all tests
uv run pytest tests/path/to/test_file.py::test_name  # Run a single test
./tools/lint         # Run ruff linter and formatter
uv run arox-coder    # Run the Coder app interactively (text UI, default)
uv run mkdocs serve  # Serve docs at http://127.0.0.1:3420
```

**Before committing**: run `./tools/lint` and `uv run pytest`.

## Configuration

Config is loaded from (in merge order):
1. `~/.config/arox/config.toml`
2. `.arox.core.config.toml` in cwd
3. App-specific defaults (e.g., `arox/apps/coder/config.toml`)
4. CLI args in dot notation (e.g., `--model_ref=openai:gpt-4o`)

`AppConfig` (Pydantic model in `arox/core/config.py`) is the central config object. It holds `AgentConfig` per agent, `ModelConfig` per model ref, `ComposerConfig` per composer, and MCP server configs.

## Architecture

### Core Abstractions

**`LLMBaseAgent`** (`arox/core/llm_base.py`): Base class for all LLM agents. Manages model inference via `pydantic_ai`, tool registration, MCP client, message history, and pre/post step hooks.

**`ChatAgent`** (`arox/core/chat.py`): Extends `LLMBaseAgent` with a conversational loop and `CommandManager` for slash commands (e.g. `/commit`, `/reset`). This is the standard agent type for user-facing agents.

**`Composer`** (`arox/core/composer.py`): Wires together a main agent, subagents, and an IO adapter into a runnable app. Subagents are registered as a `SUBAGENT` capability on the main agent. The `coder` composer is the primary example.

**`Plugin`** (`arox/core/plugin.py`): Base class for extending agents. A plugin declares:
- `tools()` — Python functions exposed to the LLM (decorated with `@tool`)
- `commands()` — slash commands for the human (decorated with `@command`)
- `history_processor()` — async hook to modify message history before LLM calls

**`Capability`** (`arox/core/capability.py`): A typed token used for loose coupling. Plugins call `agent.provide_capability(cap, impl)` and consumers call `agent.get_capability(cap)`. Defined capabilities are in `arox/plugins/capabilities.py`.

### Entry Points (Plugin System)

Components are loaded by name via setuptools entry points (defined in `pyproject.toml`):
- `arox.agents` — agent types (`chat`, `git_commit`, `compaction`)
- `arox.io_adapters` — UI adapters (`text`, `vercel_ai`, `telegram`, `feishu`)
- `arox.plugins` — plugins (`core`, `file`, `repo`, `shell`)
- `arox.hooks` — pre/post step hooks (`auto_compaction`)

To add a new plugin/agent/adapter, implement the class and register it in `pyproject.toml` then run `uv sync`.

### IO Adapters (`arox/ui/`)

Adapters abstract the UI. All agents communicate through `AgentIOInterface`. Available adapters:
- `TextIOAdapter` — rich terminal via `prompt-toolkit`
- `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
- `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Skills (`arox/core/skills.py`)

Skills are discovered from `.arox/skills/` in the workspace. They are injected into the agent's system prompt as a catalog. An `AgentConfig` can restrict which skills are available via the `skills` field.

### Built-in Apps (`arox/apps/`)

- **`coder`**: Main coding assistant. Composes a `ChatAgent` with `GitCommitAgent` and `CompactionAgent` subagents.
- **`git_commit`**: Subagent that handles `/commit` — generates commit messages and runs git.
- **`compaction`**: Subagent that compresses message history when it grows too long.
