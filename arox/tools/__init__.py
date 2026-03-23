import uuid

from pydantic_ai import RunContext
from pydantic_ai.exceptions import CallDeferred

from arox.agent_patterns.llm_base import AgentDeps


async def ask_human(ctx: RunContext["AgentDeps"], question: str) -> str:
    """
    Ask human for more information or decisions.

    Use this tool when the current task requires more input or information from the user.
    Scenarios include, but are not limited to:
    - Multiple viable options are available and user decision is required.
    - Critical information is missing and needs to be provided by the user.
    - Confirming destructive or high-risk operations (e.g., deleting databases, overwriting critical files).
    - Clarifying ambiguous requirements or instructions.
    - Requesting credentials, API keys, or sensitive data that should not be guessed.
    """
    key = str(uuid.uuid4())
    await ctx.deps.agent_io.add_tool_input_request(question, key)

    async def callback():
        return await ctx.deps.agent_io.get_tool_input_result(key)

    raise CallDeferred(metadata={"result_callback": callback})
