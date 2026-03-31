import logging

from pydantic_ai import AgentRunResult, ModelMessage
from pydantic_ai.tools import DeferredToolRequests

from arox.core.llm_base import LLMBaseAgent

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
    async def handle_task(self, task: str, main_agent: LLMBaseAgent, **kwargs) -> str:
        if main_agent.agent_session:
            main_agent.agent_session.add_event(
                "subagent_call",
                {"subagent": self.name, "task": task},
            )

        example_len = len(main_agent.example_messages)
        messages_to_compact = main_agent.message_history[example_len:]

        if not messages_to_compact:
            return "No history to compact."

        messages_before = len(main_agent.message_history)
        summary = await self.compact(messages_to_compact)

        from pydantic_ai import ModelRequest, UserPromptPart

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )

        compacted_messages = [new_request]
        main_agent.message_history = main_agent.example_messages + compacted_messages

        if main_agent.agent_session:
            from arox.core.session import _serialize_messages

            main_agent.agent_session.add_event(
                "compaction",
                {
                    "messages_before": messages_before,
                    "messages_after": len(main_agent.message_history),
                    "compacted_messages": _serialize_messages(compacted_messages),
                },
            )

        return "Conversation history compacted successfully."

    async def compact(self, messages: list[ModelMessage]) -> str:
        logger.info("Starting context compaction...")
        await self.agent_io.agent_send(
            "Context size is large. Compacting conversation history..."
        )

        from arox.core.llm_base import AgentDeps

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
        from arox.plugins.capabilities import SUBAGENT

        compaction_agent = None
        for get_subagent_func in agent.get_capability(SUBAGENT):
            compaction_agent = get_subagent_func("compaction")
            if compaction_agent:
                break

        if compaction_agent:
            await compaction_agent.handle_task("", main_agent=agent)
