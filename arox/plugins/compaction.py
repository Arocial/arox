import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic_ai import ModelMessage, ModelRequest, RunContext, UserPromptPart

from arox.core.agent_runtime import AgentRuntime, ContinueAgentRun
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, tool
from arox.core.session import CompactionEvent
from arox.plugins.slots import PERSISTENT_CONTEXT

logger = logging.getLogger(__name__)

COMPACTION_AGENT_NAME = "compaction"


@dataclass
class CompactionOutcome:
    status: Literal["compacted", "skipped", "failed"]
    messages: list[ModelMessage]
    output: str


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
        self._compaction_requested = False
        self._compaction_instructions = ""

    def _resolve_token_threshold(self) -> int | None:
        """Resolve the effective threshold for the runtime's current model.

        Model and configuration settings may change between turns, so this value
        is intentionally recomputed instead of cached on the plugin instance.
        """
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

    async def handle_compact(self, event: CompactEvent) -> str:
        runtime = self.runtime
        async with runtime.history_lock:
            messages_to_compact = runtime.session.message_history

            if not messages_to_compact:
                return "No history to compact."

            outcome = await self._compact(
                messages_to_compact, extra_instructions=event.extra_instructions
            )

            if outcome.status != "compacted":
                return outcome.output

            self._apply_compaction(
                outcome.messages,
                trigger="manual",
            )
            return outcome.output

    def _apply_compaction(
        self,
        compacted: list[ModelMessage],
        *,
        trigger: Literal["manual", "token_threshold", "tool_request"],
    ) -> None:
        """Commit replacement context and its visible marker together."""
        session = self.runtime.session
        context_id = str(uuid.uuid4())
        session.run_info.llm_context_id = context_id
        session.run_info.context_tokens = 0
        # Commit synchronously: the cause, the reset, then the replacement context.
        session.add_event(
            CompactionEvent(
                agent_name=session.agent_name,
                step_boundary=trigger == "manual",
                trigger=trigger,
                llm_context_id=context_id,
            )
        )
        session.reset_message_history()
        session.record_model_messages(compacted, run_id=context_id, context_only=True)

    async def history_processor(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        extra_instructions = ""
        trigger: Literal["token_threshold", "tool_request"] = "token_threshold"
        if self._compaction_requested:
            self._compaction_requested = False
            extra_instructions = self._compaction_instructions
            self._compaction_instructions = ""
            trigger = "tool_request"
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

        outcome = await self._compact(messages, extra_instructions=extra_instructions)
        if outcome.status != "compacted":
            return messages

        self._apply_compaction(
            outcome.messages,
            trigger=trigger,
        )
        raise ContinueAgentRun(outcome.messages)

    async def _compact(
        self, messages: list[ModelMessage], extra_instructions: str = ""
    ) -> CompactionOutcome:
        runtime = self.runtime

        if not messages:
            return CompactionOutcome("skipped", messages, "No history to compact.")

        agent_config = runtime.config.agent.get(COMPACTION_AGENT_NAME)
        if not agent_config:
            logger.warning(
                "CompactionPlugin could not find an agent config named '%s'.",
                COMPACTION_AGENT_NAME,
            )
            return CompactionOutcome(
                "skipped",
                messages,
                "Compaction agent not configured; skipping compaction.",
            )

        prompt = agent_config.task_prompt
        if not prompt:
            raise ValueError(
                "Compaction agent requires `task_prompt` to be configured."
            )
        if extra_instructions:
            prompt = f"{prompt}\n\nAdditional instructions: {extra_instructions}"

        logger.info("Compacting conversation history")

        compaction_session = await runtime.session.create_child_session(
            agent_name=COMPACTION_AGENT_NAME,
            agent_source="compaction",
            workspace=runtime.workspace,
            initial_message=prompt,
        )
        compaction_failed = False
        try:
            compaction_runtime = AgentRuntime(
                runtime.config_loader, runtime.io_adapter, compaction_session
            )
            async with compaction_runtime:
                compaction_session.record_model_messages(
                    messages, run_id=compaction_session.id, context_only=True
                )
                turn = compaction_runtime.start_message(prompt)
                result = await turn
                summary = (
                    result.output if result and isinstance(result.output, str) else ""
                )
        except Exception:
            logger.exception("Compaction agent task failed")
            summary = ""
            compaction_failed = True

        if not summary:
            logger.warning("Compaction returned no summary. Skipping.")
            return CompactionOutcome(
                "failed" if compaction_failed else "skipped",
                messages,
                "Compaction failed; skipping compaction."
                if compaction_failed
                else "Compaction returned no summary; skipping compaction.",
            )

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )
        compacted_messages: list[ModelMessage] = [new_request]

        # Add persistent context (e.g. agents.md)
        for persistent_messages in await runtime.invoke_slot(PERSISTENT_CONTEXT) or []:
            compacted_messages.extend(persistent_messages)

        return CompactionOutcome(
            "compacted",
            compacted_messages,
            "Conversation history compacted successfully.",
        )
