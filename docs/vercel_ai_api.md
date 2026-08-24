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
| `state` | Committed UI message history and selected model; sent when the IO connection is established or reconnected |
| `cmd-user-message` | An ordered user-message boundary with top-level `client_message_id` and `server_message_id`. A missing client ID is generated when `UserInput` is created; message metadata also contains the stable `user_input_id` fork anchor |
| `cmd-turn-state` | The retained turn entered or left its busy reading epoch; `busy=false` is ordered after its final output chunk |
| `cmd-session-tree` | Updated recursive session view |
| `ack` | Acknowledges a client payload and reports its status |

Client payloads:

```json
{ "cancel": true }
{ "command": { "type": "InfoEvent" } }
{ "reply": { "id": "msg-1", "role": "user", "parts": [{ "type": "text", "text": "hello", "state": "done" }] } }
```

Cancel acknowledgements report `cancelled`, `idle`, or `unavailable`. Command
acknowledgements report `ok`, `unknown`, or `invalid`.

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
