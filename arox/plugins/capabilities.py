from collections.abc import Callable
from typing import Any

from pydantic_ai import ModelMessage

from arox.core.capability import Capability

# Capability for getting project files
PROJECT_FILES = Capability[Callable[[], list[str]]](
    "project_files", "Provides a list of tracked project files"
)

# Capability for getting a subagent by name
SUBAGENT = Capability[Callable[[str], Any]](
    "subagent", "Provides access to a subagent by name"
)

# Capability for getting agent info
AGENT_INFO = Capability[Callable[[], str]](
    "agent_info", "Provides information about the agent's current state"
)

# Capability for resetting agent state
AGENT_RESET = Capability[Callable[[], None]]("agent_reset", "Resets the agent's state")

# Capability for providing persistent context that should survive compaction
PERSISTENT_CONTEXT = Capability[Callable[[], list[ModelMessage]]](
    "persistent_context", "Provides messages that should persist across compaction"
)
