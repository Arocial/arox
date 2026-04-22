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

#### `POST /api/composers/{composer_id}/agents/{agent_name}/chat`
Send a message to a specific agent and receive a streaming response (Server-Sent Events).

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, agent!"
    }
  ]
}
```

#### `GET /api/composers/{composer_id}/agents/{agent_name}/suggestions`
Get command suggestions or auto-completions for a specific agent.

**Query Parameters:**
- `command` (optional): The command to get completions for (e.g., `model`).
- `q` (optional): The current input string to filter suggestions.

#### `GET /api/composers/{composer_id}/agents/{agent_name}/history`
Get the message history for a specific agent.

### Sessions

#### `GET /api/sessions`
List all saved sessions.

#### `DELETE /api/sessions/{session_id}`
Delete a saved session.

### Health

#### `GET /api/health`
Check if the API server is running.
