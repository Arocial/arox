import logging
import uuid

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    RunContext,
    UserPromptPart,
)

from arox.core.llm_base import LLMBaseAgent
from arox.core.plugin import Plugin, command
from arox.core.session import _serialize_messages
from arox.plugins.capabilities import PERSISTENT_CONTEXT, SUBAGENT

logger = logging.getLogger(__name__)

COMPACTION_AGENT_NAME = "compaction"


class CompactionAgent(LLMBaseAgent):
    """LLMBaseAgent specialized for summarizing conversation history."""

    async def summarize(
        self, messages: list[ModelMessage], extra_instructions: str = ""
    ) -> str:
        logger.info("Starting context compaction...")
        prompt = self.agent_config.task_prompt
        if not prompt:
            raise ValueError(
                "CompactionAgent requires `task_prompt` to be set in agent config."
            )
        if extra_instructions:
            prompt = f"{prompt}\n\nAdditional instructions: {extra_instructions}"
        result = await self._run_inference(
            prompt,
            message_history=messages,
        )
        logger.info("Context compaction completed.")
        return str(result.output)


class CompactionPlugin(Plugin):
    def __init__(self, agent: LLMBaseAgent):
        super().__init__(agent)

    def _resolve_token_threshold(self) -> int | None:
        """Resolve effective token threshold for the agent's current model.

        Order of precedence: model-level `compaction_threshold`, then global
        `compaction_threshold`. Float values in (0, 1] are treated as a ratio
        of `ModelSettings.max_tokens`; otherwise the value is absolute.
        """
        agent = self.agent
        model_cfg = getattr(agent, "model_config", None)
        threshold: int | float | None = None
        if model_cfg is not None and model_cfg.compaction_threshold is not None:
            threshold = model_cfg.compaction_threshold
        else:
            threshold = getattr(agent.parsed_config, "compaction_threshold", None)
        if threshold is None:
            return None
        if isinstance(threshold, float) and 0 < threshold <= 1:
            max_tokens = (agent.model_params or {}).get("max_tokens")
            if not max_tokens:
                logger.warning(
                    "Compaction threshold %s is a ratio but model has no "
                    "max_tokens configured; skipping auto-compaction.",
                    threshold,
                )
                return None
            return int(threshold * max_tokens)
        return int(threshold)

    @command(
        "compact",
        "Compact conversation history - /compact [extra instructions]",
    )
    async def compact_command(self, name: str, arg: str):
        agent = self.agent
        messages_to_compact = list(agent.message_history)

        if not messages_to_compact:
            await agent.agent_io.agent_send("No history to compact.")
            return

        messages_before = len(agent.message_history)
        compacted_messages = await self._compact(
            messages_to_compact, extra_instructions=arg.strip()
        )

        if compacted_messages is messages_to_compact:
            return

        agent.message_history = compacted_messages
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

    async def history_processor(
        self,
        ctx: RunContext[None],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        threshold = self._resolve_token_threshold()
        if threshold is None:
            return messages

        tokens = ctx.usage.total_tokens
        if tokens > threshold:
            logger.info(
                f"Context size ({tokens} tokens) exceeds threshold ({threshold}). "
                "Triggering automatic compaction."
            )
            return await self._compact(messages)

        return messages

    def _find_compaction_agent(self) -> CompactionAgent | None:
        for get_subagent in self.agent.get_capability(SUBAGENT):
            sub = get_subagent(COMPACTION_AGENT_NAME)
            if isinstance(sub, CompactionAgent):
                return sub
        return None

    async def _compact(
        self, messages: list[ModelMessage], extra_instructions: str = ""
    ) -> list[ModelMessage]:
        agent = self.agent

        if not messages:
            return messages

        compaction_agent = self._find_compaction_agent()
        if not compaction_agent:
            await agent.agent_io.agent_send(
                "Compaction agent not configured; skipping compaction."
            )
            logger.warning(
                "CompactionPlugin could not find a CompactionAgent subagent named '%s'.",
                COMPACTION_AGENT_NAME,
            )
            return messages

        await agent.agent_io.agent_send(
            "Context size is large. Compacting conversation history..."
        )

        summary = await compaction_agent.summarize(
            messages, extra_instructions=extra_instructions
        )

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages: list[ModelMessage] = [new_request]

        # Add persistent context (e.g. agents.md)
        for get_persistent in agent.get_capability(PERSISTENT_CONTEXT):
            compacted_messages.extend(get_persistent())

        await agent.agent_io.agent_send("Conversation history compacted successfully.")
        return compacted_messages
