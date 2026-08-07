# This file provides guidance to coding agents when working with code in this repository.

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

**Before committing**: run `uv run poe agent-check`, then fix any issues.

## Architecture

### Hierarchy: App → MainAgent (with Plugins)

An **App** is a runnable process that owns one `IOAdapter` and hosts a **MainAgent**. The `MainAgent` runs the user-facing loop and is driven by `AppConfig` and `AgentConfig`. Plugins are loaded in `AgentConfig.plugins` order and receive their per-plugin settings from `AgentConfig.plugin_config`, keyed by the same plugin name used in `plugins`. Subagents are managed by the `SubagentPlugin`: its default `simple` mode exposes only `delegate_task` and waits for each one-shot delegation to complete, while `mode = "advanced"` exposes the resumable task controls (`spawn_agent`, `followup_task`, `wait_agent`, `interrupt_agent`, and `list_agents`) and surfaces active subagent runtimes through the `SUBAGENTS` slot. Subagent types are defined statically in configuration; advanced-mode tasks are represented directly by child `AgentSession`s, persist their task metadata, and instantiate ephemeral runtime agents on demand for each turn. Running tasks are cancelled and live resources are closed when the parent agent stops.

```toml
[agent.coder.plugin_config.subagent]
mode = "advanced"
```

Agent types and which agent to instantiate come from config (`arox/core/config.py`): `ConfigLoader` resolves `AppConfig` / `AgentConfig` from layered YAML plus CLI overrides, caches unchanged source files, and exposes the active snapshot through `current_config`; callers use `reload()` to pick up changed config/include/agent/skill files. Apps retain the base loader; creating a runtime agent derives an independent loader for the agent's workspace while preserving the app, profile, and CLI context. Failed runtime reloads preserve the last valid `Config` snapshot.

**Session-centric architecture** (`arox/core/session.py` and `arox/core/llm_base/agent.py`):
- **`AgentSession`** is the persistent, authoritative entity tracking:
  - Identity and hierarchy (`id`, `path`, `owner`, `children`).
  - Task metadata (`task_name`, `target`, `initial_message`, `last_message`, `last_result`, `last_error`).
  - Session usability lifecycle (`SessionStatus`: `ACTIVE`, `CLOSED`).
  - Ephemeral runtime presence via `session.has_runtime` / `session.runtime`.
  - Message history (`events`), run/token metadata (`run_info`), and persistence.
  - Agent sessions persist only the agent name and type; full `AgentConfig` is resolved dynamically from active configuration. The main agent's `AgentSession` is the top-level session; child tasks and subagents nest beneath it.
- **`LLMBaseAgent`** is an ephemeral runtime owning live/expensive resources (Pydantic AI agent, MCP clients, plugins, tools, IO channels).
  - Runtime lifecycle state machine (`AgentStatus`: `UNINITIALIZED`, `IDLE`, `RUNNING`, `STOPPED`).
  - Callers construct or load an authoritative `AgentSession` and instantiate an ephemeral runtime via `session.create_agent(config_loader, io_adapter)`.
  - Child sessions are spawned via `parent_session.create_child_session(...)`.
  - Runtime identity matches `session.id` (`agent.uuid == session.id`).
  - Agent runtime context (`async with agent:`) binds `session.runtime`, initializes plugins/tools/channels, sets `AgentStatus.IDLE`, and on exit handles cleanup, records interruption/errors to `session`, sets `AgentStatus.STOPPED`, unbinds `session.runtime`, and saves the session.
  - `step()` transitions agent status `IDLE -> RUNNING -> IDLE`.
  - `run_turn()` coordinates runtime context entry, execution hook (`execute_task()`), and result recording. `DelegatableAgent.run_task()` delegates to `run_turn()`.
  - Subagent task display states (such as `running`, `completed`, `interrupted`, `error`, `pending`, `idle`, `closed`) are derived on-the-fly from live runtime/task state and session result/error data.
  - Subsequent turns reuse the persistent `AgentSession` and reconstruct a fresh runtime via `session.create_agent(config_loader, io_adapter)`.
- `SessionManager` coordinates with `SessionStore` (default `FileSessionStore`) to persist sessions to disk with debouncing and age-based cleanup.
- Resuming is done via the `session_id` passed to the App. The `CorePlugin` provides the `/fork` command.

### IO system

IO is split into two layers: per-agent channels and app-level adapters.

- **Per-agent channel** (`arox/core/io.py`): `create_io_channel()` returns a pair of `IOEndpoint` instances backed by two in-memory streams. Every agent holds its own `agent_io: IOEndpoint` and uses `send` / `receive` to talk to the UI — both the main agent and each subagent have independent channels, so their output can be routed/rendered separately. `RequestEvent` / `ReplyEvent` provide request/reply correlation on top of `send` / `receive`.
- **App-level adapter** (`arox/ui/`, base `AbstractIOAdapter`): one adapter per App. It registers hosts (agents), consumes each agent's adapter-side `IOEndpoint`, and renders events to the concrete UI. Available adapters:
  - `TextIOAdapter` — rich terminal via `prompt-toolkit`
  - `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
  - `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Agents

**Types**
- **`LLMBaseAgent`** (`arox/core/llm_base/agent.py`): base class for all LLM agents. Owns model inference via `pydantic_ai`, tool registration, MCP clients, message history, and pre/post step hooks.
- **`MainAgent`** (`arox/core/llm_base/agent.py`): abstract subclass that an App's main agent must extend; hosts the user-driven run loop entry point.
- **`ChatAgent`** (`arox/core/chat.py`): concrete `MainAgent` adding a conversational loop and `CommandManager` for slash commands (e.g. `/model`, `/reset`). Standard choice for user-facing agents.

**Extension points**
- **`Plugin`** (`arox/core/plugin.py`): primary extension unit. A plugin declares:
  - methods decorated with `@tool` — Python functions exposed to the LLM; the decorator's `enabled` condition can make exposure depend on plugin configuration
  - `commands()` — slash commands for the human (`@command`)
  - `history_processor()` — async hook to modify message history before LLM calls
- **`Slot`** (`arox/core/slot.py`): typed token for loose coupling between plugins/agents, used for both pull and push patterns. Producers call `agent.provide_slot(slot, impl)`; consumers pull or push notifications with `await agent.invoke_slot(slot, ...)`. Built-in slots are in `arox/plugins/slots.py` (e.g. `SUBAGENT`, `PERSISTENT_CONTEXT`, `AGENT_RESET`).
- **Skills**: Discovered automatically during configuration loading (`arox/core/config.py`) from `~/.config/arox/skills/` and `.agents/skills/` directories in the workspace and global paths. They are injected into the agent's system prompt as an XML catalog. `AgentConfig.skills` restricts which are visible. `AgentConfig.default_skills` allows loading the content of specific skills directly into the system prompt.
- **MCP**: each agent can connect to MCP servers through its `pydantic_ai` client, exposing remote tools alongside local ones.
