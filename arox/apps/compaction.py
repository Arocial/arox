import logging

from pydantic_ai import AgentRunResult, ModelMessage
from pydantic_ai.tools import DeferredToolRequests

from arox.agent_patterns.llm_base import LLMBaseAgent

logger = logging.getLogger(__name__)

COMPACTION_PROMPT = """
Provide a detailed prompt for continuing our conversation above. Focus on information that would be helpful for continuing the conversation, including what we did, what we're doing, which files we're working on, and what we're going to do next. The summary that you construct will be used so that another agent can read it and continue the work. When constructing the summary, try to stick to this template:

---
## Goal
[What goal(s) is the user trying to accomplish?]

## Instructions
- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it so next agent can continue using it]

## Discoveries
[What notable things were learned during this conversation that would be useful for the next agent to know when continuing the work]

## Accomplished
[What work has been completed, what work is still in progress, and what work is left?]

## Relevant files / directories
[Construct a structured list of relevant files that have been read, edited, or created that pertain to the task at hand. If all the files in a directory are relevant, include the path to the directory.]
---
"""


class CompactionAgent(LLMBaseAgent):
    async def compact(self, messages: list[ModelMessage]) -> str:
        logger.info("Starting context compaction...")
        await self.agent_io.agent_send(
            "Context size is large. Compacting conversation history..."
        )

        from arox.agent_patterns.llm_base import AgentDeps

        result = await self.pydantic_agent.run(
            COMPACTION_PROMPT,
            message_history=messages,
            deps=AgentDeps(agent_io=self.agent_io),
        )

        logger.info("Context compaction completed.")
        return str(result.output)


async def auto_compaction_hook(
    agent: LLMBaseAgent,
    input_content: str | None,
    result: AgentRunResult[DeferredToolRequests | str] | None,
) -> None:
    if not result:
        return
    usage = result.usage()
    if usage and usage.request_tokens and usage.request_tokens > 100000:
        logger.info(
            f"Context size ({usage.request_tokens} tokens) exceeds threshold. Triggering automatic compaction."
        )
        from arox import commands

        await commands.CompactionCommand(agent).execute("compact", "")
