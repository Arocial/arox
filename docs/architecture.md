# Architecture

Arox is organized around a clear runtime hierarchy, a split IO system, and a small set of extension points on top of a common agent base. This document walks through each layer.

## Hierarchy: App → AgentSession + SessionRunner

An **App** is a runnable process (e.g. `arox-coder`) that owns a single **IO adapter** and activates an `AgentSession` through a `SessionRunner`.

- Exactly one user-facing session driven by `ServeRunner` and `ChatServeDriver`.
- Zero or more **subagents** — specialized agents run as resumable tasks by the `SubagentPlugin`. The main agent can spawn, wait for, interrupt, inspect, and continue these tasks through callable tools.
- A resolved `AppConfig` and `AgentConfig` (from `arox/core/config.py`), which names the main agent, its subagents, and their per-agent configuration.

The runner creates the common `AgentRuntime`, initializes its plugins and IO resources, and closes it through `start_runtime()` / `stop_runtime()` or async context management. `TaskRunner.run()` owns a resumable turn task; `ServeRunner.run()` owns the long-lived serve-loop task, while `ChatServeDriver` separately owns the current interaction task.

### Session management

Session handling is a core capability provided by `arox/core/session.py`:

- **`AgentSession`** is the persistent source of truth for identity, task metadata, events, and segmented message history. Events retain audit metadata, while `message_history` and `archived_message_histories` store the messages used by the runtime and historical forks. The main agent's `AgentSession` is the top-level session for a run (the one addressed by `session_id`); subagents keep their own `AgentSession`s nested beneath it. Every `AgentRuntime` is constructed with an `AgentSession`.
- **`llm_context_id`** is a UUID tracking the current LLM context window, passed to providers (e.g. via headers) to leverage server-side caching. Compaction rolls it forward, signaling a new context.
- **`SessionManager`** and **`SessionStore`** (default `FileSessionStore`) persist sessions as JSON with age-based cleanup. Sessions are loaded and provided to runtimes upon initialization, and saved on exit; resume by passing `session_id` to the App. The `CorePlugin` focuses on user commands like `/fork`.

## IO system

IO is split into two layers: a per-agent channel and an app-level adapter.

### Per-agent channel

Every runtime holds its own **`IOEndpoint`** (`runtime.agent_ep`), created by `create_io_channel()` in `arox/core/io.py` together with a paired adapter-side `IOEndpoint`. Every endpoint exposes its owning `IOHost` through `endpoint.host`, so adapter code can recover the runtime without embedding it in individual events. Endpoints are backed by a pair of in-memory streams and expose `send` / `receive`. Because the main runtime and each subagent runtime have independent channels, their output can be routed, rendered, or stored independently. `RequestEvent` / `ReplyEvent` add request/reply correlation: passing a `RequestEvent` to `send` awaits the matching `ReplyEvent` and returns it.

### App-level adapter

One **`AbstractIOAdapter`** (`arox/ui/`) is instantiated per App. The adapter:

- Consumes each agent's matching adapter-side `IOEndpoint`.
- Renders events to the concrete UI.

Built-in adapters:

- **`TextIOAdapter`** — rich terminal via `prompt-toolkit`.
- **`VercelStreamIOAdapter`** — web frontend via Vercel AI SDK (FastAPI/SSE).
- **`TelegramIOAdapter`** — Telegram bot.
- **`FeishuIOAdapter`** — Feishu (Lark) bot.

## Agents

### Types

- **`AgentRuntime`** (`arox/core/agent_runtime.py`) — the concrete ephemeral LLM runtime. Owns model inference via `pydantic_ai`, tool registration (local + MCP), plugins, IO resources, and turn hooks.
- **`TaskRunner`** (`arox/core/runner.py`) — owns the latest resumable turn task returned by `run()`.
- **`ServeRunner`** (`arox/core/runner.py`) — owns the serve-loop task; `ChatServeDriver` owns and can cancel only the current interaction.
- **`ChatServeDriver`** (`arox/core/chat.py`) — implements the chat request/reply protocol.

### Extension points

- **Plugins** (`arox/core/plugin.py`): the primary unit of extension. A plugin bundles:
    - methods decorated with `@tool` — Python functions exposed to the LLM. Arox also supports **MCP** tools via `fastmcp`, registered alongside local tools.
    - `commands()` — slash / control commands for the human. Plugins override `commands()` to return `CommandSpec(event_cls, handler, completer)` bindings that `CommandManager` dispatches. Commands run locally without calling the LLM, saving time and tokens.
    - `history_processor()` — async hook that modifies message history before each LLM call.
- **Slots** (`arox/core/slot.py`): typed tokens for loose coupling, used for both pull and push patterns. Producers call `runtime.provide_slot(slot, impl)`; consumers pull or push notifications with `await runtime.invoke_slot(slot, ...)`. Built-in slots live in `arox/plugins/slots.py` (e.g. `PROJECT_FILES`, `PERSISTENT_CONTEXT`).
- **Skills**: Discovered automatically during configuration loading (`arox/core/config.py`) from `~/.config/arox/skills/` and `.agents/skills/` directories in the workspace and global paths. They are injected into the agent's system prompt as an XML catalog. `AgentConfig.skills` restricts which skills are visible to a given agent.

## Data flow

1. **User input** arrives at the IO adapter and is forwarded over the main agent's `IOEndpoint`.
2. **Command check**: the IO adapter tests whether the input is a slash command and, if so, executes it locally without calling the LLM.
3. **LLM inference**: otherwise the message is appended to history (an `agent_step` event) and sent to the LLM via `pydantic_ai`.
4. **Tool execution**: tool calls (local, MCP, or a subagent) are dispatched and their results fed back to the LLM.
5. **Response**: the final text is streamed back through the agent's IO channel and rendered by the adapter.
