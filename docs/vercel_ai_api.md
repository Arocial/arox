# API Reference

Arox provides a REST API that is compatible with the Vercel AI SDK, allowing you to build web frontends for your agents.

## Endpoints

### Composers

#### `POST /api/composers`
Create a new composer instance.

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
  "id": "composer-uuid",
  "workspace": "/path/to/workspace",
  "main_agent": "main",
  "subagents": ["coder", "planner"]
}
```

#### `GET /api/composers`
List all running composer instances.

**Response:**
```json
[
  {
    "id": "composer-uuid",
    "workspace": "/path/to/workspace",
    "main_agent": "main",
    "subagents": ["coder", "planner"]
  }
]
```

#### `DELETE /api/composers/{composer_id}`
Stop and delete a composer instance.

### Agent Interactions

The following endpoints are per-agent, meaning you must specify both the `composer_id` and the `agent_name` (which can be the `main_agent` or one of the `subagents` from the `ComposerInfo` response).

#### `WS /api/composers/{composer_id}/agents/{agent_name}/ws`
Full-duplex WebSocket for async interaction with an agent. This connection is long-lived across multiple steps — sending input and receiving events are decoupled.

**Server → Client messages** (JSON, one object per frame)

Each frame is a message conforming to the [Vercel AI SDK data stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) (`text-*`, `reasoning-*`, `tool-*`, `finish`, etc.), delivered as a plain JSON object per WS frame instead of SSE `data:` lines.

Arox adds a few non-standard frames on the same channel:

| type | fields | meaning |
|---|---|---|
| `data-input-request` | `data` | agent is waiting for user input; `data` carries `req_id`, `deferred_tools`, `normal_input`, `exception_input` |
| `step-done` | — | current step fully drained; next step may follow on the same connection |
| `ack` | `status` | acknowledgment of a client-sent message (see below) |

**Client → Server messages** (JSON)

```json
// submit a reply to a pending data-input-request
{ "reply": { "req_id": "<uuid from data-input-request>", "normal_input": { "user_input": "hello" } } }

// resume after reconnecting: re-send the currently pending data-input-request, if any
{ "resume": true }

// cancel the agent's currently running foreground step
{ "cancel": true }
```

The `reply` object's `req_id` MUST match the `req_id` carried in the `data-input-request` it answers; remaining fields (`deferred_tools`, `normal_input`, `exception_input`) mirror the matching fields of the request. The server responds with `{"type": "ack", "status": "ok" | "cancelled" | "no_req_id" | "noop"}`.

The connection stays open until either side closes it. Closing the client disconnects the stream but does **not** cancel any in-flight step — send `{"cancel": true}` first if needed.

#### `GET /api/composers/{composer_id}/agents/{agent_name}/suggestions`
Get command suggestions or auto-completions for a specific agent.

**Query Parameters:**
- `command` (optional): The command to get completions for (e.g., `model`).
- `q` (optional): The current input string to filter suggestions.

#### `GET /api/composers/{composer_id}/agents/{agent_name}/state`
Get the current state for a specific agent: message history plus any pending input request.

**Response:**
```json
{
  "history": [ /* Vercel AI UI messages */ ]
}
```

To recover any pending input prompt after reconnecting, send `{"resume": true}` over the WebSocket — the server will re-emit the currently pending `data-input-request`, if any.

### Sessions

#### `GET /api/sessions`
List all saved sessions.

#### `DELETE /api/sessions/{session_id}`
Delete a saved session.

### Health

#### `GET /api/health`
Check if the API server is running.
