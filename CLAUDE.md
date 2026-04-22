#his file provides guidance to coding agents(e.g. Claude Code) when working with code in this repository.

## Development Commands

```bash
uv sync              # Install dependencies
uv run pytest        # Run all tests
./tools/lint         # Run ruff linter and formatter
uv run arox-coder    # Run the Coder app interactively (text UI, default)
uv run mkdocs serve  # Serve docs at http://127.0.0.1:3420
```

**Before committing**: run `./tools/lint` `uv run pytest` and fix.

## Architecture

### Core Abstractions

**`LLMBaseAgent`** (`arox/core/llm_base.py`): Base class for all LLM agents. Manages model inference via `pydantic_ai`, tool registration, MCP client, message history, and pre/post step hooks.

**`ChatAgent`** (`arox/core/chat.py`): Extends `LLMBaseAgent` with a conversational loop and `CommandManager` for slash commands (e.g. `/model`, `/reset`). This is the standard agent type for user-facing agents.

**`Composer`** (`arox/core/composer.py`): Wires together a main agent, subagents, and an IO adapter into a runnable app. Subagents are registered as a `SUBAGENT` capability on the main agent. The `coder` composer is the primary example.

**`Plugin`** (`arox/core/plugin.py`): Base class for extending agents. A plugin declares:
- `tools()` — Python functions exposed to the LLM (decorated with `@tool`)
- `commands()` — slash commands for the human (decorated with `@command`)
- `history_processor()` — async hook to modify message history before LLM calls

**`Capability`** (`arox/core/capability.py`): A typed token used for loose coupling. Plugins call `agent.provide_capability(cap, impl)` and consumers call `agent.get_capability(cap)`. Defined capabilities are in `arox/plugins/capabilities.py`.

### IO Adapters (`arox/ui/`)

Adapters abstract the UI. All agents communicate through `AgentIOEndpoint`. Available adapters:
- `TextIOAdapter` — rich terminal via `prompt-toolkit`
- `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
- `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Skills (`arox/core/skills.py`)

Skills are discovered from `.arox/skills/` in the workspace. They are injected into the agent's system prompt as a catalog. An `AgentConfig` can restrict which skills are available via the `skills` field.
