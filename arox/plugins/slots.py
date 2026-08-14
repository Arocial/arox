from collections.abc import Callable

from pydantic_ai import ModelMessage

from arox.core.slot import ListSlot

# Slot for getting project files
PROJECT_FILES = ListSlot[Callable[[], list[str]], list[str]](
    "project_files", "Provides a list of tracked project files"
)

# Slot for getting agent info
AGENT_INFO = ListSlot[Callable[[], str], str](
    "agent_info", "Provides information about the agent's current state"
)

# Slot for providing persistent context that should survive compaction
PERSISTENT_CONTEXT = ListSlot[Callable[[], list[ModelMessage]], list[ModelMessage]](
    "persistent_context", "Provides messages that should persist across compaction"
)

# Slot for providing additional system prompt fragments
SYSTEM_PROMPT = ListSlot[Callable[[], str], str](
    "system_prompt", "Provides additional system prompt fragments"
)
