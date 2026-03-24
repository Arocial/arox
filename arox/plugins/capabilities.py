from collections.abc import Callable
from typing import Any

from arox.agent_patterns.capability import Capability

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
