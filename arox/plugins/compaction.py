import logging
import uuid

from pydantic_ai import (
    AgentRunResult,
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolRequests

from arox.core.llm_base import LLMBaseAgent
from arox.core.plugin import Plugin, command
from arox.core.session import _serialize_messages
from arox.plugins.capabilities import SUBAGENT

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

DEFAULT_TOKEN_THRESHOLD = 100000

COMPACTION_AGENT_NAME = "compaction"


class CompactionAgent(LLMBaseAgent):
    """LLMBaseAgent specialized for summarizing conversation history."""

    async def summarize(self, messages: list[ModelMessage]) -> str:
        logger.info("Starting context compaction...")
        result = await self._run_inference(
            COMPACTION_PROMPT,
            message_history=messages,
        )
        logger.info("Context compaction completed.")
        return str(result.output)


class CompactionPlugin(Plugin):
    def __init__(self, agent: LLMBaseAgent):
        super().__init__(agent)
        self.token_threshold = DEFAULT_TOKEN_THRESHOLD
        agent.add_post_step_hook(self._auto_compact_hook)

    @command("compact", "Compact conversation history - /compact")
    async def compact_command(self, name: str, arg: str):
        await self._compact()

    async def _auto_compact_hook(
        self,
        agent: LLMBaseAgent,
        input_content: str | None,
        result: AgentRunResult[DeferredToolRequests | str] | None,
    ) -> None:
        if not result:
            return
        usage = result.usage()
        tokens = getattr(usage, "input_tokens", None) if usage else None
        if tokens is None and usage is not None:
            tokens = getattr(usage, "request_tokens", None)
        if tokens and tokens > self.token_threshold:
            logger.info(
                f"Context size ({tokens} tokens) exceeds threshold. Triggering automatic compaction."
            )
            await self._compact()

    def _find_compaction_agent(self) -> CompactionAgent | None:
        for get_subagent in self.agent.get_capability(SUBAGENT):
            sub = get_subagent(COMPACTION_AGENT_NAME)
            if isinstance(sub, CompactionAgent):
                return sub
        return None

    async def _compact(self) -> None:
        agent = self.agent
        example_len = len(agent.example_messages)
        messages_to_compact = agent.message_history[example_len:]

        if not messages_to_compact:
            await agent.agent_io.agent_send("No history to compact.")
            return

        compaction_agent = self._find_compaction_agent()
        if not compaction_agent:
            await agent.agent_io.agent_send(
                "Compaction agent not configured; skipping compaction."
            )
            logger.warning(
                "CompactionPlugin could not find a CompactionAgent subagent named '%s'.",
                COMPACTION_AGENT_NAME,
            )
            return

        await agent.agent_io.agent_send(
            "Context size is large. Compacting conversation history..."
        )

        messages_before = len(agent.message_history)
        summary = await compaction_agent.summarize(messages_to_compact)

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages = [new_request]
        agent.message_history = agent.example_messages + compacted_messages
        agent.llm_context_id = str(uuid.uuid4())

        if agent.agent_session:
            agent.agent_session.add_event(
                "compaction",
                {
                    "messages_before": messages_before,
                    "messages_after": len(agent.message_history),
                    "compacted_messages": _serialize_messages(compacted_messages),
                    "llm_context_id": agent.llm_context_id,
                },
            )

        await agent.agent_io.agent_send("Conversation history compacted successfully.")
