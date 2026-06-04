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


# --- Push slots (handlers invoked by agent.notify) ---

# Emitted after the agent's message history and llm_context_id are cleared.
AGENT_RESET = Slot[Callable[[], Any]](
    "agent_reset",
    "The agent's conversational state was reset",
    aggregator=ResultAggregator.DISCARD,
)
