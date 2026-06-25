import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    RunContext,
    UserPromptPart,
)

from arox.core.llm_base import DelegatableAgent, LLMBaseAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.plugins.slots import DELEGATE_TO_SUBAGENT, PERSISTENT_CONTEXT

logger = logging.getLogger(__name__)

COMPACTION_AGENT_NAME = "compaction"


@dataclass(kw_only=True)
class CompactEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("compact",)
    description: ClassVar[str] = (
        "Compact conversation history - /compact [extra instructions]"
    )

    extra_instructions: str = ""

    @classmethod
    def from_slash(cls, name, arg):
        return cls(extra_instructions=(arg or "").strip())


class CompactionAgent(DelegatableAgent):
    """DelegatableAgent specialized for summarizing conversation history."""

    async def run_task(self, task: str) -> str | None:
        return await self.summarize(self.message_history, extra_instructions=task)

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
        self._last_total_tokens = 0
        self._cached_threshold_resolved = False
        self._cached_threshold_value = None

    def _resolve_token_threshold(self) -> int | None:
        """Resolve effective token threshold for the agent's current model.

        Order of precedence: model-level `compaction_threshold`, then global
        `compaction_threshold`. Float values in (0, 1] are treated as a ratio
        of `ModelSettings.max_tokens`; otherwise the value is absolute.
        """
        if self._cached_threshold_resolved:
            return self._cached_threshold_value

        agent = self.agent
        model_cfg = agent.model_config
        threshold: int | float | None = None
        if model_cfg is not None and model_cfg.compaction_threshold is not None:
            threshold = model_cfg.compaction_threshold
        else:
            threshold = agent.parsed_config.compaction_threshold

        resolved_val = None
        if threshold is not None:
            if isinstance(threshold, float) and 0 < threshold <= 1:
                max_tokens = (agent.model_params or {}).get("max_tokens")
                if max_tokens:
                    resolved_val = int(threshold * max_tokens)
            else:
                resolved_val = int(threshold)

        logger.info("Resolved compaction token threshold: %s", resolved_val)

        self._cached_threshold_value = resolved_val
        self._cached_threshold_resolved = True
        return resolved_val

    def commands(self):
        return [CommandSpec(CompactEvent, self.handle_compact)]

    async def handle_compact(self, event: CompactEvent):
        agent = self.agent
        messages_to_compact = list(agent.message_history)

        if not messages_to_compact:
            await agent.agent_io.send("No history to compact.")
            return

        compacted_messages = await self._compact(
            messages_to_compact, extra_instructions=event.extra_instructions
        )

        if compacted_messages is messages_to_compact:
            return

        agent.message_history = compacted_messages
        await self._record_compaction(compacted_messages, True)

    async def _record_compaction(
        self, compacted: list[ModelMessage], step_boundary: bool
    ) -> None:
        """Record a ``compaction`` event.

        ``step_boundary`` indicates if the compaction is inside one agent step.
        """
        agent = self.agent
        agent.run_info.llm_context_id = str(uuid.uuid4())
        agent.session.record_compaction(
            compacted,
            step_boundary,
            agent.run_info.llm_context_id,
        )

    async def history_processor(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        threshold = self._resolve_token_threshold()
        if threshold is None:
            return messages

        context_tokens = self.agent.run_info.context_tokens

        if context_tokens <= threshold:
            return messages

        logger.info(
            f"Last request size ({context_tokens} tokens) exceeds threshold ({threshold}). "
            "Triggering automatic compaction."
        )
        compacted = await self._compact(messages)
        if compacted is messages:
            return messages

        await self._record_compaction(compacted, False)
        from pydantic_ai._agent_graph import _first_new_message_index

        if ctx.run_id:
            self.agent.new_message_index = _first_new_message_index(
                messages, ctx.run_id, resumed_request=None
            )
        return compacted

    async def _compact(
        self, messages: list[ModelMessage], extra_instructions: str = ""
    ) -> list[ModelMessage]:
        agent = self.agent

        if not messages:
            return messages

        agent_config = agent.parsed_config.agent.get(COMPACTION_AGENT_NAME)
        if not agent_config:
            await agent.agent_io.send(
                "Compaction agent not configured; skipping compaction."
            )
            logger.warning(
                "CompactionPlugin could not find a CompactionAgent config named '%s'.",
                COMPACTION_AGENT_NAME,
            )
            return messages

        await agent.agent_io.send(
            "Context size is large. Compacting conversation history..."
        )

        def setup_compaction_subagent(subagent: Any) -> None:
            subagent.message_history = messages.copy()

        summary = await agent.invoke_slot(
            DELEGATE_TO_SUBAGENT,
            COMPACTION_AGENT_NAME,
            extra_instructions,
            on_subagent_created=setup_compaction_subagent,
        )

        if not summary:
            logger.warning("Compaction returned no summary. Skipping.")
            return messages

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages: list[ModelMessage] = [new_request]

        # Add persistent context (e.g. agents.md)
        for persistent_messages in await agent.invoke_slot(PERSISTENT_CONTEXT) or []:
            compacted_messages.extend(persistent_messages)

        await agent.agent_io.send("Conversation history compacted successfully.")
        return compacted_messages
