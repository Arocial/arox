import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic_ai import ModelMessage, ModelRequest, RunContext, UserPromptPart

from arox.core.agent_runtime import AgentRuntime
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, tool
from arox.core.runner import TaskRunner
from arox.plugins.slots import PERSISTENT_CONTEXT

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


class CompactionPlugin(Plugin):
    def __init__(self, runtime: AgentRuntime):
        super().__init__(runtime)
        self._last_total_tokens = 0
        self._cached_threshold_resolved = False
        self._cached_threshold_value = None
        self._compaction_requested = False
        self._compaction_instructions = ""

    def _resolve_token_threshold(self) -> int | None:
        """Resolve effective token threshold for the runtime's current model.

        Order of precedence: model-level `compaction_threshold`, then global
        `compaction_threshold`. Float values in (0, 1] are treated as a ratio
        of `ModelSettings.max_tokens`; otherwise the value is absolute.
        """
        if self._cached_threshold_resolved:
            return self._cached_threshold_value

        runtime = self.runtime
        model_cfg = runtime.model_config
        threshold: int | float | None = None
        if model_cfg is not None and model_cfg.compaction_threshold is not None:
            threshold = model_cfg.compaction_threshold
        else:
            threshold = runtime.config.compaction_threshold

        resolved_val = None
        if threshold is not None:
            if isinstance(threshold, float) and 0 < threshold <= 1:
                max_tokens = (runtime.model_params or {}).get("max_tokens")
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

    @tool(sequential=True)
    def compact(self, instruction: str = "") -> str:
        """Request conversation history compaction before the next model request.

        Args:
            instruction: Optional instructions for how to summarize the history.
        """
        self._compaction_requested = True
        self._compaction_instructions = instruction.strip()
        return "Conversation history compaction requested."

    async def handle_compact(self, event: CompactEvent):
        runtime = self.runtime
        messages_to_compact = list(runtime.message_history)

        if not messages_to_compact:
            await runtime.agent_ep.send("No history to compact.")
            return

        compacted_messages = await self._compact(
            messages_to_compact, extra_instructions=event.extra_instructions
        )

        if compacted_messages is messages_to_compact:
            return

        await self._record_compaction(
            compacted_messages, True, previous_messages=messages_to_compact
        )

    async def _record_compaction(
        self,
        compacted: list[ModelMessage],
        step_boundary: bool,
        *,
        previous_messages: list[ModelMessage],
    ) -> None:
        """Record a ``compaction`` event.

        ``step_boundary`` indicates if the compaction is inside one runtime turn.
        """
        runtime = self.runtime
        runtime.run_info.llm_context_id = str(uuid.uuid4())
        runtime.session.record_compaction(
            compacted,
            step_boundary,
            runtime.run_info.llm_context_id,
            previous_messages=previous_messages,
        )

    async def history_processor(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        extra_instructions = ""
        if self._compaction_requested:
            self._compaction_requested = False
            extra_instructions = self._compaction_instructions
            self._compaction_instructions = ""
        else:
            threshold = self._resolve_token_threshold()
            if threshold is None:
                return messages

            context_tokens = self.runtime.run_info.context_tokens
            if context_tokens <= threshold:
                return messages

            logger.info(
                f"Last request size ({context_tokens} tokens) exceeds threshold ({threshold}). "
                "Triggering automatic compaction."
            )

        compacted = await self._compact(messages, extra_instructions=extra_instructions)
        if compacted is messages:
            return messages

        await self._record_compaction(compacted, False, previous_messages=messages)
        from pydantic_ai._agent_graph import _first_new_message_index

        if ctx.run_id:
            self.runtime.new_message_index = _first_new_message_index(
                messages,
                ctx.run_id,
                resumed_request=None,
                resumed_request_index=None,
            )
        return compacted

    async def _compact(
        self, messages: list[ModelMessage], extra_instructions: str = ""
    ) -> list[ModelMessage]:
        runtime = self.runtime

        if not messages:
            return messages

        agent_config = runtime.config.agent.get(COMPACTION_AGENT_NAME)
        if not agent_config:
            await runtime.agent_ep.send(
                "Compaction agent not configured; skipping compaction."
            )
            logger.warning(
                "CompactionPlugin could not find an agent config named '%s'.",
                COMPACTION_AGENT_NAME,
            )
            return messages

        prompt = agent_config.task_prompt
        if not prompt:
            raise ValueError(
                "Compaction agent requires `task_prompt` to be configured."
            )
        if extra_instructions:
            prompt = f"{prompt}\n\nAdditional instructions: {extra_instructions}"

        await runtime.agent_ep.send(
            "Context size is large. Compacting conversation history..."
        )

        compaction_session = await runtime.session.create_child_session(
            agent_name=COMPACTION_AGENT_NAME,
            agent_source="compaction",
            workspace=runtime.workspace,
            initial_message=prompt,
            last_message=prompt,
        )
        try:
            async with TaskRunner(
                compaction_session, runtime.config_loader, runtime.io_adapter
            ) as runner:
                compaction_runtime = runner.runtime
                assert compaction_runtime is not None
                compaction_runtime.message_history = messages.copy()
                result = await runner.run(prompt)
                summary = (
                    result.output if result and isinstance(result.output, str) else ""
                )
        except Exception:
            logger.exception("Compaction agent task failed")
            summary = ""

        if not summary:
            logger.warning("Compaction returned no summary. Skipping.")
            return messages

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages: list[ModelMessage] = [new_request]

        # Add persistent context (e.g. agents.md)
        for persistent_messages in await runtime.invoke_slot(PERSISTENT_CONTEXT) or []:
            compacted_messages.extend(persistent_messages)

        await runtime.agent_ep.send("Conversation history compacted successfully.")
        return compacted_messages
