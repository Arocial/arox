# API Reference

Arox provides a REST API that is compatible with the Vercel AI SDK, allowing you to build web frontends for your agents.

## Endpoints

### Agents

#### `POST /api/agents`
Create a new agent instance.

**Request Body:**
```json
{
  "workspace": "/path/to/workspace", // Optional
  "session_id": "session-uuid"       // Optional
}
```

**Response:**
```json
{
  "id": "agent-uuid",
  "workspace": "/path/to/workspace",
  "main_agent": "main",
  "subagents": ["coder", "planner"]
}
```

#### `GET /api/agents`
List all running agent instances.

**Response:**
```json
[
  {
    "id": "agent-uuid",
    "workspace": "/path/to/workspace",
    "main_agent": "main",
    "subagents": ["coder", "planner"]
  }
]
```

#### `DELETE /api/agents/{agent_id}`
Stop and delete an agent instance.

### Agent Interactions

The following endpoints are per-agent, meaning you must specify both the `agent_id` and the `agent_name` (which can be the `main_agent` or one of the `subagents` from the `AgentInfo` response).

#### `WS /api/agents/{agent_id}/{agent_name}/ws`
Full-duplex WebSocket for async interaction with an agent. This connection is long-lived across multiple steps — sending input and receiving events are decoupled.

**Server → Client messages** (JSON, one object per frame)

Each frame is a message conforming to the [Vercel AI SDK data stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) (`text-*`, `reasoning-*`, `tool-*`, `finish`, etc.), delivered as a plain JSON object per WS frame instead of SSE `data:` lines.

Arox adds a few non-standard frames on the same channel:

| type | fields | meaning |
|---|---|---|
| `data-input-request` | `data` | agent is waiting for user input; `data` carries `req_id`, `deferred_tools`, `normal_input`, `exception_input` |
| `data-user-turn` | `eventIndex`, `messageId?` | a user-turn anchor was just recorded in the agent session; `eventIndex` is the absolute event index usable as `ForkEvent.event_index`. `messageId` echoes the `client_message_id` carried on the matching `reply`, when present |
| `step-done` | — | current step fully drained; next step may follow on the same connection |
| `ack` | `status` | acknowledgment of a client-sent message (see below) |

**Client → Server messages** (JSON)

```json
// submit a reply to a pending data-input-request
{ "reply": { "req_id": "<uuid from data-input-request>", "normal_input": { "user_input": "hello" } } }

// invoke a slash / control command without going through the LLM
{ "command": { "type": "SetModelEvent", "model_ref": "claude-opus-4-7" } }

// resume after reconnecting: re-send the currently pending data-input-request, if any
{ "resume": true }

// cancel the agent's currently running foreground step
{ "cancel": true }
```

The `reply` object's `req_id` MUST match the `req_id` carried in the `data-input-request` it answers; remaining fields (`deferred_tools`, `normal_input`, `exception_input`) mirror the matching fields of the request. An optional `client_message_id` may be included to identify the UI message that produced this reply — when set, the server stores it on the resulting `user_input` session event and echoes it in the subsequent `data-user-turn` frame as `messageId`, so the client can map UI messages back to absolute event indices without relying on ordering.

The `command` payload is a structured `CommandEvent`: `type` is the event class name (e.g. `SetModelEvent`, `InfoEvent`, `ResetEvent`, `AgentCallEvent`, `FileAddEvent`, `CompactEvent`, `AddFileListEvent`), and the remaining fields populate that event's dataclass. The server runs the command locally and streams any reply text back as ordinary text frames; the ack carries `{"status": "ok" | "unknown_command", "output": "..."}`.

The server responds to every client message with `{"type": "ack", "status": "ok" | "cancelled" | "unknown_command" | "no_req_id" | "noop", ...}`.

The connection stays open until either side closes it. Closing the client disconnects the stream but does **not** cancel any in-flight step — send `{"cancel": true}` first if needed.

#### `GET /api/agents/{agent_id}/{agent_name}/suggestions`
Get command suggestions or auto-completions for a specific agent.

**Query Parameters:**
- `command` (optional): The slash name to get argument completions for (e.g., `model`, `add`). When omitted, returns the list of all registered slash commands.
- `q` (optional): Filter / current input string. Without `command`, filters slash names; with `command`, used as the in-progress argument fragment.

**Response:** `{"items": [{"id", "value", "label", "description"}, ...]}`. When listing slash commands, `description` is taken from the `CommandEvent` subclass's `description` ClassVar.

#### `GET /api/agents/{agent_id}/{agent_name}/state`
Get the current state for a specific agent: message history plus any pending input request.

**Response:**
```json
{
  "history": [ /* Vercel AI UI messages */ ],
  "model": "claude-opus-4-7"
}
```

`model` is the current `provider_model` on the agent (the same value `/info` reports), or `null` if unset.

`user_turn_anchors` is a map from UI `message_id` to the absolute event index of its `user_input` session event. Pass a value as `ForkEvent.event_index` to fork at that turn. The server populates the map from `client_message_id` stored on each `user_input` event; for legacy events that predate the field, entries fall back to the corresponding user message's id by chronological order. New entries arrive live as `data-user-turn` frames on the agent WebSocket — each frame carries both `eventIndex` and (when the client supplied it) `messageId`.

To recover any pending input prompt after reconnecting, send `{"resume": true}` over the WebSocket — the server will re-emit the currently pending `data-input-request`, if any.

### Sessions

#### `GET /api/sessions`
List all saved sessions.

#### `DELETE /api/sessions/{session_id}`
Delete a saved session.

### Health

#### `GET /api/health`
Check if the API server is running.
