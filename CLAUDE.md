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

### Hierarchy: App → Composer → Agents

An **App** is a runnable process that owns one `IOAdapter` and hosts one or more **Composers**. Each `Composer` (`arox/core/composer.py`) wires together a **main agent** plus zero or more **subagents** against a single workspace, driven by `ComposerConfig`. The main agent runs the user-facing loop; subagents are exposed to it via the `SUBAGENT` capability so it can delegate.

Agent types and which agent to instantiate come from config (`arox/core/config.py`): `AppConfig` / `ComposerConfig` / `AgentConfig` are resolved by `load_config` from layered YAML plus CLI overrides.

**Session management** lives at the composer layer (`arox/core/session.py`):
- `ComposerSession` aggregates per-agent `AgentSession` entries (message history + metadata) under one session id.
- `SessionStore` (default `FileSessionStore`) persists sessions to disk with an age-based cleanup.
- On `Composer.run()`, sessions are restored into each agent; on exit they are saved back. Resuming is done via the `session_id` passed to `Composer`.

### IO system

IO is split into two layers: per-agent channels and app-level adapters.

- **Per-agent channel** (`arox/core/io.py`): `create_io_channel()` returns an `(AgentIOEndpoint, AdapterIOEndpoint)` pair backed by two in-memory streams. Every agent holds its own `agent_io: AgentIOEndpoint` and uses `agent_send` / `agent_receive` to talk to the UI — both the main agent and each subagent have independent channels, so their output can be routed/rendered separately.
- **App-level adapter** (`arox/ui/`, base `AbstractIOAdapter`): one adapter per App, shared across composers. It registers composers, consumes each agent's `AdapterIOEndpoint`, and renders events to the concrete UI. Available adapters:
  - `TextIOAdapter` — rich terminal via `prompt-toolkit`
  - `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
  - `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Agents

**Types**
- **`LLMBaseAgent`** (`arox/core/llm_base.py`): base class for all LLM agents. Owns model inference via `pydantic_ai`, tool registration, MCP clients, message history, and pre/post step hooks.
- **`MainAgent`** (`arox/core/llm_base.py`): abstract subclass that a `Composer`'s main agent must extend; hosts the user-driven run loop entry point.
- **`ChatAgent`** (`arox/core/chat.py`): concrete `MainAgent` adding a conversational loop and `CommandManager` for slash commands (e.g. `/model`, `/reset`). Standard choice for user-facing agents.

**Extension points**
- **`Plugin`** (`arox/core/plugin.py`): primary extension unit. A plugin declares:
  - `tools()` — Python functions exposed to the LLM (`@tool`)
  - `commands()` — slash commands for the human (`@command`)
  - `history_processor()` — async hook to modify message history before LLM calls
- **`Capability`** (`arox/core/capability.py`): typed token for loose coupling between plugins/agents. Producers call `agent.provide_capability(cap, impl)`; consumers call `agent.get_capability(cap)`. Built-in capabilities are in `arox/plugins/capabilities.py` (e.g. `SUBAGENT`).
- **Skills** (`arox/core/skills.py`): discovered from `.arox/skills/` in the workspace and injected into the agent's system prompt as a catalog. `AgentConfig.skills` restricts which are visible.
- **MCP**: each agent can connect to MCP servers through its `pydantic_ai` client, exposing remote tools alongside local ones.
- **Hooks**: `pre_step_hooks` / `post_step_hooks` from `AgentConfig` are loaded via entry points and attached by the composer around each inference step.
