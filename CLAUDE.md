#his file provides guidance to coding agents(e.g. Claude Code) when working with code in this repository.

## Development Commands

```bash
uv sync              # Install dependencies
uv run pytest        # Run all tests
uv run ruff format   # Format code
uv run ruff check --fix  # Lint and auto-fix (config in pyproject.toml)
uv run ty check      # Type check
uv run arox-coder    # Run the Coder app interactively (text UI, default)
uv run mkdocs serve  # Serve docs at http://127.0.0.1:3420
```

**Before committing**: run `uv run ruff format && uv run ruff check --fix && uv run ty check`, and `uv run pytest`, then fix any issues.

## Architecture

### Hierarchy: App → MainAgent (with Plugins)

An **App** is a runnable process that owns one `IOAdapter` and hosts a **MainAgent**. The `MainAgent` runs the user-facing loop and is driven by `AppConfig` and `AgentConfig`. Subagents are managed by the `SubagentPlugin`, which instantiates them and exposes them to the main agent as callable tools (and via the `SUBAGENT` slot) so it can delegate tasks directly.

Agent types and which agent to instantiate come from config (`arox/core/config.py`): `AppConfig` / `AgentConfig` are resolved by `load_config` from layered YAML plus CLI overrides.

**Session management** is handled by the `SessionPlugin` (`arox/plugins/session.py`):
- The main agent's `AgentSession` (message history + metadata) is the top-level session for a run; subagents keep their own `AgentSession`s nested beneath it.
- `SessionStore` (default `FileSessionStore`) persists sessions to disk with an age-based cleanup.
- On agent start, sessions are restored into each agent; on exit they are saved back. Resuming is done via the `session_id` passed to the App.

### IO system

IO is split into two layers: per-agent channels and app-level adapters.

- **Per-agent channel** (`arox/core/io.py`): `create_io_channel()` returns a pair of `IOEndpoint` instances backed by two in-memory streams. Every agent holds its own `agent_io: IOEndpoint` and uses `send` / `receive` to talk to the UI — both the main agent and each subagent have independent channels, so their output can be routed/rendered separately. `RequestEvent` / `ReplyEvent` provide request/reply correlation on top of `send` / `receive`.
- **App-level adapter** (`arox/ui/`, base `AbstractIOAdapter`): one adapter per App. It registers hosts (agents), consumes each agent's adapter-side `IOEndpoint`, and renders events to the concrete UI. Available adapters:
  - `TextIOAdapter` — rich terminal via `prompt-toolkit`
  - `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
  - `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Agents

**Types**
- **`LLMBaseAgent`** (`arox/core/llm_base.py`): base class for all LLM agents. Owns model inference via `pydantic_ai`, tool registration, MCP clients, message history, and pre/post step hooks.
- **`MainAgent`** (`arox/core/llm_base.py`): abstract subclass that an App's main agent must extend; hosts the user-driven run loop entry point.
- **`ChatAgent`** (`arox/core/chat.py`): concrete `MainAgent` adding a conversational loop and `CommandManager` for slash commands (e.g. `/model`, `/reset`). Standard choice for user-facing agents.

**Extension points**
- **`Plugin`** (`arox/core/plugin.py`): primary extension unit. A plugin declares:
  - `tools()` — Python functions exposed to the LLM (`@tool`)
  - `commands()` — slash commands for the human (`@command`)
  - `history_processor()` — async hook to modify message history before LLM calls
- **`Slot`** (`arox/core/slot.py`): typed token for loose coupling between plugins/agents, used for both pull and push patterns. Producers call `agent.provide_slot(slot, impl)`; consumers pull or push notifications with `await agent.invoke_slot(slot, ...)`. Built-in slots are in `arox/plugins/slots.py` (e.g. `SUBAGENT`, `PERSISTENT_CONTEXT`, `AGENT_RESET`).
- **Skills** (`arox/core/skills.py`): discovered from `.arox/skills/` in the workspace and injected into the agent's system prompt as a catalog. `AgentConfig.skills` restricts which are visible.
- **MCP**: each agent can connect to MCP servers through its `pydantic_ai` client, exposing remote tools alongside local ones.
