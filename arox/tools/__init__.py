import uuid

from pydantic_ai import RunContext
from pydantic_ai.exceptions import CallDeferred

from arox.agent_patterns.llm_base import AgentDeps


async def ask_human(ctx: RunContext["AgentDeps"], question: str) -> str:
    """Ask human for more information or decisions."""
    key = str(uuid.uuid4())
    await ctx.deps.agent_io.add_tool_input_request(question, key)

    async def callback():
        return await ctx.deps.agent_io.get_tool_input_result(key)

    raise CallDeferred(metadata={"result_callback": callback})
