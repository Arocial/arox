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

### Hierarchy: App → AgentSession + SessionRunner

An **App** is a runnable process that owns one `IOAdapter` and activates an `AgentSession` through a `SessionRunner`. `ServingRunner` owns the user-facing serve loop and `TaskRunner` owns resumable subagent turns. Both create the same `LLMBaseAgent` runtime. Plugins are loaded in `AgentConfig.plugins` order and receive their settings from `AgentConfig.plugin_config`. The `SubagentPlugin` exposes synchronous delegation in `simple` mode and resumable task controls in `advanced` mode.

```toml
[agent.coder.plugin_config.subagent]
mode = "advanced"
```

`ConfigLoader` resolves `AppConfig` / `AgentConfig` from layered YAML plus CLI overrides, caches unchanged source files, and exposes the active snapshot through `current_config`; callers use `reload()` to pick up changed config/include/agent/skill files. Agent configuration no longer selects a runtime class.

**Session-centric architecture** (`arox/core/session.py` and `arox/core/llm_base/agent.py`):
- **`AgentSession`** is the persistent, authoritative entity tracking:
  - Identity and hierarchy (`id`, `path`, `owner`, `children`).
  - Task metadata (`task_name`, `target`, `initial_message`, `last_message`, `result`, `error`).
  - Ephemeral execution presence via `session.runner`; runners are not persisted.
  - Message history is persisted as segments on the session and is the runtime source of truth. Events store only audit metadata; user turns are located through the existing `server_message_id` carried in `UserInput.input_content`. Compaction archives the processor's original messages, including the current user request, so historical forks can locate and slice the appropriate messages without replaying events or duplicating message payloads or IDs.
  - Agent sessions persist only the agent name; full `AgentConfig` is resolved dynamically. The user-facing `AgentSession` is the top-level session; child tasks nest beneath it.
- **`LLMBaseAgent`** is an ephemeral runtime owning live/expensive resources (Pydantic AI agent, MCP clients, plugins, tools, IO channels).
  - It executes one turn through `step()` and does not own asyncio tasks.
  - A `SessionRunner` creates, enters, and closes the runtime; startup is serialized through the session and failed initialization rolls back the runner binding.
  - Child sessions are spawned via `parent_session.create_child_session(...)`.
  - Runtime identity matches `session.id` (`agent.uuid == session.id`).
  - `TaskRunner.run()` starts one turn; `wait()` and `cancel()` manage it while retaining the runtime for follow-ups.
  - `ServingRunner` separately owns a long-lived serve task and the current turn task. `cancel_turn()` preserves the serve loop; `stop()` closes both tasks and the runtime.
  - `ChatServeDriver` implements the concrete request/reply protocol used by `ServingRunner`.
- `SessionManager` coordinates with `SessionStore` (default `FileSessionStore`) and maintains one authoritative in-process Session identity map keyed by full path. Its tree API (`resolve`, `list_roots`, `children_of`, `walk`, `find`, `stop_tree`, `delete_tree`, and `remove_child`) transparently prefers cached/live instances over storage, while manager shutdown stops every live root tree before flushing pending saves.
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
- **`LLMBaseAgent`** (`arox/core/llm_base/agent.py`): the single LLM runtime. Owns inference, plugins, tools, MCP clients, IO channels, and single-turn `step()` execution.
- **`TaskRunner`** (`arox/core/runner.py`): manages resumable task turns.
- **`ServingRunner`** (`arox/core/runner.py`): manages the long-lived interaction loop and its current turn.
- **`ChatServeDriver`** (`arox/core/chat.py`): implements the chat request/reply loop.

**Extension points**
- **`Plugin`** (`arox/core/plugin.py`): primary extension unit. A plugin declares:
  - methods decorated with `@tool` — Python functions exposed to the LLM; the decorator's `enabled` condition can make exposure depend on plugin configuration
  - `commands()` — slash commands for the human (`@command`)
  - `history_processor()` — async hook to modify message history before LLM calls
- **`Slot`** (`arox/core/slot.py`): typed token for loose coupling between plugins/agents, used for both pull and push patterns. Producers call `agent.provide_slot(slot, impl)`; consumers pull or push notifications with `await agent.invoke_slot(slot, ...)`. Built-in slots are in `arox/plugins/slots.py` (e.g. `SUBAGENTS`, `PERSISTENT_CONTEXT`).
- **Skills**: Discovered automatically during configuration loading (`arox/core/config.py`) from `~/.config/arox/skills/` and `.agents/skills/` directories in the workspace and global paths. They are injected into the agent's system prompt as an XML catalog. `AgentConfig.skills` restricts which are visible. `AgentConfig.default_skills` allows loading the content of specific skills directly into the system prompt.
- **MCP**: each agent can connect to MCP servers through its `pydantic_ai` client, exposing remote tools alongside local ones.
