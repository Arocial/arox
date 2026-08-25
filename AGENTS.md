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

### Hierarchy: App → AgentSession + AgentRuntime

An **App** is a runnable process that owns one `IOAdapter` and activates an `AgentRuntime` for an `AgentSession`. The runtime owns the current `Turn`, while `AgentIOEndpoint` dispatches inbound user input. Plugins are loaded in `AgentConfig.plugins` order and receive their settings from `AgentConfig.plugin_config`. The `SubagentPlugin` exposes synchronous delegation in `simple` mode and resumable task controls in `advanced` mode.

```toml
[agent.coder.plugin_config.subagent]
mode = "advanced"
```

`ConfigLoader` resolves `AppConfig` / `AgentConfig` from layered YAML plus CLI overrides, caches unchanged source files, and exposes the active snapshot through `current_config`; callers use `reload()` to pick up changed config/include/agent/skill files. Agent configuration no longer selects a runtime class.

**Session-centric architecture** (`arox/core/session.py` and `arox/core/agent_runtime.py`):
- **`AgentSession`** is the persistent, authoritative entity tracking:
  - Identity and hierarchy (`id`, `path`, `owner`, `children`).
  - Task metadata (`task_name`, `target`, `initial_message`). Task results and errors are available through the active runtime's `Turn` rather than persisted on the session.
  - Ephemeral execution presence via `session.runtime`; runtimes are not persisted.
  - Message history is persisted as segments on the session and is the runtime source of truth for LLM context. Session events form the UI timeline: user-input events carry the original input, command request/completion events carry command replies, and step events reference stable IDs stored in `ModelMessage.metadata`. `build_io_snapshot()` projects those events together with active and archived message segments, so commands are visible after resume without entering the model context. User turns are located through the existing `server_message_id` carried in `UserInput.input_content`. Compaction archives the processor's original messages, including the current user request, so historical forks can locate and slice the appropriate messages without replaying events or duplicating model payloads.
  - Agent sessions persist only the agent name; full `AgentConfig` is resolved dynamically. The user-facing `AgentSession` is the top-level session; child tasks nest beneath it.
- **`AgentRuntime`** is an ephemeral runtime owning live/expensive resources (Pydantic AI agent, MCP clients, plugins, tools, IO channels).
  - It executes one input through a retained `Turn`, which wraps the active `asyncio.Task`.
  - It supports async context management; startup is serialized through the session and failed initialization rolls back the runtime binding.
  - Child sessions are spawned via `parent_session.create_child_session(...)`.
  - Runtime identity matches `session.id` (`runtime.uuid == session.id`).
  - `AgentRuntime.accept_input()` dispatches slash commands or creates a `Turn`; callers can await, shield, time out, or cancel its task while the runtime remains available for follow-ups.
  - `AgentIOEndpoint` dispatches adapter-originated `UserInput` directly to the runtime. There is no separate serve task. `AgentRuntime.cancel_turn()` cancels the active turn while preserving the runtime, and `close()` cancels execution before closing resources.
- `SessionManager` coordinates with `SessionStore` (default `FileSessionStore`) and maintains one authoritative in-process Session identity map keyed by full path. Its tree API (`resolve`, `list_roots`, `children_of`, `walk`, `find`, `stop_tree`, `delete_tree`, and `remove_child`) transparently prefers cached/live instances over storage, while manager shutdown stops every live root tree before flushing pending saves.
- Resuming is done via the `session_id` passed to the App. The `CorePlugin` provides the `/fork` command.

### IO system

IO is split into two layers: per-agent channels and app-level adapters.

- **Per-agent channel** (`arox/core/io.py`): every runtime holds its own `agent_ep: AgentIOEndpoint` and uses `send` / `receive` to talk to the UI. Adapter-originated events are dispatched to registered handlers by the agent endpoint's receive loop. Both the main runtime and each subagent runtime have independent channels, so their output can be routed/rendered separately. `RequestEvent` / `ReplyEvent` remain available when explicit request/reply correlation is needed.
- **App-level adapter** (base `AbstractIOAdapter` in `arox/core/io.py`; Chat implementations in `arox/apps/chat/io_adapters/`): one adapter per App. Each `IOHost` registers itself with the adapter for its active lifetime and owns the adapter-side consumer for its paired `IOEndpoint`; the adapter renders those events to the concrete UI. Available adapters:
  - `TextIOAdapter` — rich terminal via `prompt-toolkit`
  - `VercelStreamIOAdapter` — web frontend via Vercel AI SDK (FastAPI/SSE)
  - `TelegramIOAdapter`, `FeishuIOAdapter` — chat bots

### Agents

**Types**
- **`AgentRuntime`** (`arox/core/agent_runtime.py`): the single LLM runtime. Owns inference, plugins, tools, MCP clients, IO channels, and single-turn `run_turn()` execution.
- **`Turn`** (`arox/core/turn.py`): wraps one input execution and exposes its task, result, error, waiting, and cancellation APIs.

**Extension points**
- **`Plugin`** (`arox/core/plugin.py`): primary extension unit. A plugin declares:
  - methods decorated with `@tool` — Python functions exposed to the LLM; the decorator's `enabled` condition can make exposure depend on plugin configuration
  - `commands()` — slash commands for the human (`@command`)
  - `history_processor()` — async hook to modify message history before LLM calls
- **`Slot`** (`arox/core/slot.py`): typed token for loose coupling between plugins/agents, used for both pull and push patterns. Producers call `runtime.provide_slot(slot, impl)`; consumers pull or push notifications with `await runtime.invoke_slot(slot, ...)`. Built-in slots are in `arox/plugins/slots.py` (e.g. `PROJECT_FILES`, `PERSISTENT_CONTEXT`).
- **Skills**: Discovered automatically during configuration loading (`arox/core/config.py`) from `~/.config/arox/skills/` and `.agents/skills/` directories in the workspace and global paths. They are injected into the agent's system prompt as an XML catalog. `AgentConfig.skills` restricts which are visible. `AgentConfig.default_skills` allows loading the content of specific skills directly into the system prompt.
- **MCP**: each agent can connect to MCP servers through its `pydantic_ai` client, exposing remote tools alongside local ones.
