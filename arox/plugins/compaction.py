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
        agent.add_post_step_hook(self._auto_compact_hook)

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
        await self._compact(extra_instructions=arg.strip())

    async def _auto_compact_hook(
        self,
        agent: LLMBaseAgent,
        input_content: str | None,
        result: AgentRunResult[DeferredToolRequests | str] | None,
    ) -> None:
        if not result:
            return
        threshold = self._resolve_token_threshold()
        if threshold is None:
            return
        usage = result.usage()
        tokens = getattr(usage, "input_tokens", None) if usage else None
        if tokens is None and usage is not None:
            tokens = getattr(usage, "request_tokens", None)
        if tokens and tokens > threshold:
            logger.info(
                f"Context size ({tokens} tokens) exceeds threshold ({threshold}). "
                "Triggering automatic compaction."
            )
            await self._compact()

    def _find_compaction_agent(self) -> CompactionAgent | None:
        for get_subagent in self.agent.get_capability(SUBAGENT):
            sub = get_subagent(COMPACTION_AGENT_NAME)
            if isinstance(sub, CompactionAgent):
                return sub
        return None

    async def _compact(self, extra_instructions: str = "") -> None:
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
        summary = await compaction_agent.summarize(
            messages_to_compact, extra_instructions=extra_instructions
        )

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages = [new_request]

        # Add persistent context (e.g. agents.md)
        for get_persistent in agent.get_capability(PERSISTENT_CONTEXT):
            compacted_messages.extend(get_persistent())

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
