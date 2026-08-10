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
  "result": null,
  "error": null,
  "children": []
}
```

## Session-node interaction

A root session and all of its subagent sessions are addressed through the same
node routes. `target_session_id` must belong to the tree identified by
`root_session_id`.

### `WS /api/sessions/{root_session_id}/nodes/{target_session_id}/ws`

Open the Vercel AI event stream for an active session node. An inactive target
is rejected. Disconnecting does not cancel an in-flight turn.

Server-specific frames include:

| type | meaning |
|---|---|
| `cmd-input-request` | Runtime is waiting for user input |
| `cmd-user-turn` | A user-turn anchor was recorded |
| `cmd-session-tree` | Updated recursive session view |
| `step-done` | Current turn was fully drained |
| `stream-close` | Close the current UI message stream |
| `ack` | Acknowledges a client payload |

Client payloads:

```json
{ "resume": true }
{ "cancel": true }
{ "command": { "type": "InfoEvent" } }
{ "reply": { "id": "msg-1", "role": "user", "content": "hello", "metadata": { "custom": { "chatInputEventResult": { "req_id": "request-id" } } } } }
```

### `GET /api/sessions/{root_session_id}/nodes/{target_session_id}/state`

Return Vercel UI message history directly from the target Session. This endpoint
works while the runtime is inactive. The model is read from the active runtime
when present and otherwise derived from Session/config data.

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
