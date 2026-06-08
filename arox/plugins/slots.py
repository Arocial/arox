from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelMessage

from arox.core.slot import DiscardSlot, FirstSlot, ListSlot

if TYPE_CHECKING:
    from arox.core.llm_base import LLMBaseAgent

# Slot for getting project files
PROJECT_FILES = ListSlot[Callable[[], list[str]], list[str]](
    "project_files", "Provides a list of tracked project files"
)

SUBAGENTS = FirstSlot[Callable[[], list["LLMBaseAgent"]], list["LLMBaseAgent"]](
    "subagents",
    "Provides the list of subagents managed by the agent",
)

# Slot for getting agent info
AGENT_INFO = ListSlot[Callable[[], str], str](
    "agent_info", "Provides information about the agent's current state"
)

# Slot for providing persistent context that should survive compaction
PERSISTENT_CONTEXT = ListSlot[Callable[[], list[ModelMessage]], list[ModelMessage]](
    "persistent_context", "Provides messages that should persist across compaction"
)


# --- Push slots (handlers invoked by agent.notify) ---

# Emitted after the agent's message history and llm_context_id are cleared.
AGENT_RESET = DiscardSlot[Callable[[], Any]](
    "agent_reset",
    "The agent's conversational state was reset",
)
