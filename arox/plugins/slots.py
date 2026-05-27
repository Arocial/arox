from collections.abc import Callable
from typing import Any

from pydantic_ai import ModelMessage

from arox.core.slot import Slot

# Slot for getting project files
PROJECT_FILES = Slot[Callable[[], list[str]]](
    "project_files", "Provides a list of tracked project files"
)

# Slot for getting a subagent by name
SUBAGENT = Slot[Callable[[str], Any]](
    "subagent", "Provides access to a subagent by name"
)

# Slot for listing the subagents that can be delegated tasks
DELEGATABLE_SUBAGENTS = Slot[Callable[[], list[Any]]](
    "delegatable_subagents",
    "Provides the list of subagents that can be delegated tasks",
)

# Slot for getting agent info
AGENT_INFO = Slot[Callable[[], str]](
    "agent_info", "Provides information about the agent's current state"
)

# Slot for resetting agent state
AGENT_RESET = Slot[Callable[[], None]]("agent_reset", "Resets the agent's state")

# Slot for providing persistent context that should survive compaction
PERSISTENT_CONTEXT = Slot[Callable[[], list[ModelMessage]]](
    "persistent_context", "Provides messages that should persist across compaction"
)

# Returns a list of all active agents (MainAgent + Subagents)
ALL_AGENTS = "all_agents"
