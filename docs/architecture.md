# Architecture

Arox is organized around a clear runtime hierarchy, a split IO system, and a small set of extension points on top of a common agent base. This document walks through each layer.

## Hierarchy: App → Composer → Agents

An **App** is a runnable process (e.g. `arox-coder`) that owns a single **IO adapter** and hosts one or more **Composers**.

A **`Composer`** (`arox/core/composer.py`) assembles a working agent system against a workspace:

- Exactly one **main agent** — the user-facing entry point, must subclass `MainAgent` (typically a `ChatAgent`).
- Zero or more **subagents** — specialized agents the main agent can delegate to. They are exposed as callable tools (and through the `SUBAGENT` slot) on the main agent.
- A resolved `ComposerConfig` (from `arox/core/config.py`), which names the main agent, its subagents, and their per-agent configuration.

The composer drives lifecycle: it constructs agents (looked up by entry-point name from `AgentConfig.type`), attaches pre/post step hooks, enters their async contexts, restores session state, then runs `main_agent.run()`.

### Session management

Session handling lives at the composer layer (`arox/core/session.py`), one layer above individual agents:

- **`ComposerSession`** aggregates per-agent `AgentSession` entries plus composer-level metadata under a single session id.
- **`AgentSession`** is event-sourced: instead of storing a static message list, it stores a sequence of `SessionEvent`s (`agent_step`, `compaction`, `reset`, …) and rebuilds `message_history` by replay.
- **`llm_context_id`** is a UUID tracking the current LLM context window, passed to providers (e.g. via headers) to leverage server-side caching. A `reset` or `compaction` event rolls it forward, signaling a new context.
- **`SessionStore`** (default `FileSessionStore`) persists sessions as JSON with age-based cleanup. `Composer.run()` restores on entry and saves on exit; resume by passing `session_id` to the composer.

## IO system

IO is split into two layers: a per-agent channel and an app-level adapter.

### Per-agent channel

Every agent holds its own **`IOEndpoint`** (`agent.agent_io`), created by `create_io_channel()` in `arox/core/io.py` together with a paired adapter-side `IOEndpoint`. Endpoints are backed by a pair of in-memory streams and expose `send` / `receive`. Because the main agent and each subagent each have their own channel, their output can be routed, rendered, or stored independently. `RequestEvent` / `ReplyEvent` add request/reply correlation: passing a `RequestEvent` to `send` awaits the matching `ReplyEvent` and returns it.

### App-level adapter

One **`AbstractIOAdapter`** (`arox/ui/`) is instantiated per App and shared across all composers in it. The adapter:

- Registers composers via `register_composer`.
- Consumes each agent's matching adapter-side `IOEndpoint`.
- Renders events to the concrete UI.

Built-in adapters:

- **`TextIOAdapter`** — rich terminal via `prompt-toolkit`.
- **`VercelStreamIOAdapter`** — web frontend via Vercel AI SDK (FastAPI/SSE).
- **`TelegramIOAdapter`** — Telegram bot.
- **`FeishuIOAdapter`** — Feishu (Lark) bot.

## Agents

### Types

- **`LLMBaseAgent`** (`arox/core/llm_base.py`) — base class for all LLM agents. Owns model inference via `pydantic_ai`, tool registration (local + MCP), message history, and pre/post step hooks.
- **`MainAgent`** (`arox/core/llm_base.py`) — abstract subclass a composer's main agent must extend; defines the top-level run loop entry point.
- **`ChatAgent`** (`arox/core/chat.py`) — concrete `MainAgent` that adds a conversational loop and a `CommandManager` for slash commands (`/model`, `/reset`, …). Standard choice for user-facing agents.

### Extension points

- **Plugins** (`arox/core/plugin.py`): the primary unit of extension. A plugin bundles:
    - `tools()` — Python functions exposed to the LLM (`@tool`). Arox also natively supports **MCP** tools via `fastmcp`, registered alongside local tools.
    - `commands()` — slash / control commands for the human. Plugins override `commands()` to return `CommandSpec(event_cls, handler, completer)` bindings that `CommandManager` dispatches. Commands run locally without calling the LLM, saving time and tokens.
    - `history_processor()` — async hook that modifies message history before each LLM call.
- **Slots** (`arox/core/slot.py`): typed tokens for loose coupling. Producers call `agent.provide_slot(slot, impl)`; consumers call `agent.get_slot(slot)`. Built-in slots live in `arox/plugins/slots.py` (e.g. `SUBAGENT`, `PERSISTENT_CONTEXT`).
- **Skills** (`arox/core/skills.py`): discovered from `.arox/skills/` in the workspace and injected into the agent's system prompt as a catalog. `AgentConfig.skills` restricts which skills are visible to a given agent.
- **Hooks**: `pre_step_hooks` / `post_step_hooks` declared in `AgentConfig` are resolved via entry points and attached around each inference step.

## Data flow

1. **User input** arrives at the IO adapter and is forwarded over the main agent's `IOEndpoint`.
2. **Command check**: the `ChatAgent` tests whether the input is a slash command and, if so, executes it locally without calling the LLM.
3. **LLM inference**: otherwise the message is appended to history (an `agent_step` event) and sent to the LLM via `pydantic_ai`.
4. **Tool execution**: tool calls (local, MCP, or a subagent) are dispatched and their results fed back to the LLM.
5. **Response**: the final text is streamed back through the agent's IO channel and rendered by the adapter.
