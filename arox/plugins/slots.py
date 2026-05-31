from collections.abc import Callable
from typing import Any

from pydantic_ai import ModelMessage

from arox.core.slot import ResultAggregator, Slot

# Slot for getting project files
PROJECT_FILES = Slot[Callable[[], list[str]]](
    "project_files", "Provides a list of tracked project files"
)

SUBAGENTS = Slot[Callable[[], list[Any]]](
    "subagents",
    "Provides the list of subagents managed by the agent",
    aggregator=ResultAggregator.FIRST,
)

# Slot for getting agent info
AGENT_INFO = Slot[Callable[[], str]](
    "agent_info", "Provides information about the agent's current state"
)

# Slot for providing persistent context that should survive compaction
PERSISTENT_CONTEXT = Slot[Callable[[], list[ModelMessage]]](
    "persistent_context", "Provides messages that should persist across compaction"
)

# Slot for getting the current LLM context ID
LLM_CONTEXT_ID = Slot[Callable[[], str]](
    "llm_context_id", "Provides the current LLM context ID"
)

# Slot for getting the current agent session
AGENT_SESSION = Slot[Callable[[], Any]](
    "agent_session",
    "Provides the current agent session",
    aggregator=ResultAggregator.FIRST,
)

# Slot for getting the agent's session store.
SESSION_STORE = Slot[Callable[[], Any]](
    "session_store",
    "Provides the agent's session store",
    aggregator=ResultAggregator.FIRST,
)

# Slot for configuring the agent's session before it starts.
# Payload: (session_id, owner_path, session_store, agent_session=None). A given
# ``agent_session`` is adopted as-is, skipping the load by id.
SET_SESSION = Slot[Callable[..., Any]](
    "set_session",
    "Sets the agent's session id, owner path and session store",
    aggregator=ResultAggregator.DISCARD,
)

# --- Push slots (handlers invoked by agent.notify) ---

# Emitted after the agent's message history and llm_context_id are cleared.
AGENT_RESET = Slot[Callable[[], Any]](
    "agent_reset",
    "The agent's conversational state was reset",
    aggregator=ResultAggregator.DISCARD,
)

# Emitted after a successful inference step. Payload: (result,).
AGENT_STEP = Slot[Callable[..., Any]](
    "agent_step",
    "An agent step completed successfully",
    aggregator=ResultAggregator.DISCARD,
)

# Emitted when an inference step fails. Payload: (messages,).
AGENT_STEP_FAILURE = Slot[Callable[..., Any]](
    "agent_step_failure", "An agent step failed", aggregator=ResultAggregator.DISCARD
)

# Emitted when a slash command is parsed. Payload: (command, arg).
AGENT_COMMAND = Slot[Callable[..., Any]](
    "agent_command", "A slash command was parsed", aggregator=ResultAggregator.DISCARD
)

# Emitted when user input is received. Payload: (text,).
USER_INPUT = Slot[Callable[..., Any]](
    "user_input", "User input was received", aggregator=ResultAggregator.DISCARD
)

# Emitted when the run loop catches an error. Payload: (error,).
AGENT_ERROR = Slot[Callable[..., Any]](
    "agent_error", "An error occurred during a run", aggregator=ResultAggregator.DISCARD
)

# Generic record channel. Payload: (event_type, data).
RECORD_EVENT = Slot[Callable[..., Any]](
    "record_event", "A custom event was recorded", aggregator=ResultAggregator.DISCARD
)
