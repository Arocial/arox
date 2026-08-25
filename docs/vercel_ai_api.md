# Vercel AI API

Arox exposes persisted `AgentSession` objects as its API resources. An active
session has a runtime; stopping a session releases that runtime without deleting
the persisted conversation.

## Session lifecycle

### `GET /api/sessions`

List saved root sessions, including their child-session trees. The `active`
field is derived from runtime presence and is not persisted.

### `POST /api/sessions`

Create and start a new root session.

```json
{ "workspace": "/optional/workspace" }
```

### `GET /api/sessions/{session_id}`

Return one root session and its descendants.

### `POST /api/sessions/{session_id}/start`

Start the runtime and interaction loop for a saved root session. Starting an
already active session returns HTTP 409.

### `POST /api/sessions/{session_id}/stop`

Stop the interaction loop and all runtimes in the session tree while retaining
the persisted session.

### `DELETE /api/sessions/{session_id}`

Stop the full session tree and delete it from the session store.

Session responses use this recursive shape:

```json
{
  "id": "session-uuid",
  "path": ["session-uuid"],
  "agent_name": "coder",
  "created_at": "...",
  "updated_at": "...",
  "workspace": "/workspace",
  "metadata": {},
  "active": true,
  "task_name": null,
  "target": null,
  "children": []
}
```

## Session-node interaction

A root session and all of its subagent sessions are addressed through the same
node routes. `target_session_id` must belong to the tree identified by
`root_session_id`.

### `WS /api/sessions/{root_session_id}/nodes/{target_session_id}/ws`

Open the Vercel AI event stream for an active session node. After accepting the
connection, the server sends a `state` frame containing the committed Vercel UI
message history and the selected model, then replays any events emitted since
that snapshot before streaming new events. Internal Arox messages are omitted
from the history.

Only one WebSocket may be connected to a session node at a time. A newer
connection closes the previous one with code `4000`; this lets a reconnect pick
up from the latest snapshot plus cached events without duplicating persisted
history. Connecting to an inactive node closes the socket with code `4004`.
Disconnecting does not cancel an in-flight turn.

Server-specific frames include:

| type | meaning |
|---|---|
| `state` | One ordered, discriminated history containing UI messages and completed commands, plus the selected model; sent when the IO connection is established or reconnected |
| `cmd-client-input` | A normalized client input changed lifecycle state. Message payloads are emitted when `started`; command payloads when `accepted` |
| `cmd-command-completed` | A normalized command input finished dispatch, with its status and optional output or error |
| `cmd-turn-state` | The retained turn entered or left its busy reading epoch; `busy=false` is ordered after its final output chunk |
| `cmd-session-tree` | Updated recursive session view |

Client payloads:

```json
{ "cancel": true }
{ "command": { "type": "InfoEvent" }, "client_message_id": "client-command-1" }
{ "reply": { "id": "msg-1", "role": "user", "parts": [{ "type": "text", "text": "hello", "state": "done" }] } }
```

The runtime does not send acknowledgement frames. A `cmd-client-input` frame
contains `client_message_id`, `server_message_id`, and a discriminated `payload`.
Message payloads carry `status: "started"` plus their Vercel UI message when model
processing starts. Command payloads carry `status: "accepted"` plus the normalized
command. A slash-prefixed message may therefore be normalized into a command,
whose result subsequently arrives in `cmd-command-completed`.

The former HTTP `.../state` endpoint is no longer exposed. State bootstrap and
recovery are part of the WebSocket stream, so clients must start the target
runtime before connecting.

### `GET /api/sessions/{root_session_id}/nodes/{target_session_id}/suggestions`

Return slash-command completions. This endpoint requires an active runtime
because completion providers belong to the runtime command manager.

Query parameters:

- `command`: optional slash-command name.
- `q`: current completion text.

## Health

### `GET /api/health`

Return server health. When `AROX_API_TOKEN` is configured, all HTTP and
WebSocket endpoints except health require a bearer token or `token` query
parameter.
