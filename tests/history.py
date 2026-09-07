import uuid
from collections.abc import Sequence
from typing import Literal

from pydantic_ai.messages import ModelMessage

from arox.core.session import (
    AgentSession,
    ContextResetEvent,
    ModelMessageEvent,
    user_input_ids_on,
)
from arox.plugins.compaction import ApplyCompaction, CompactionEvent


def record_messages(session: AgentSession, messages: Sequence[ModelMessage]) -> None:
    run_id = uuid.uuid4().hex
    for sequence, message in enumerate(messages):
        session.record(
            ModelMessageEvent(run_id=run_id, sequence=sequence, message=message)
        )


def context_resets(session: AgentSession) -> list[ContextResetEvent]:
    return [entry for entry in session.journal if isinstance(entry, ContextResetEvent)]


def contains_input(messages: Sequence[ModelMessage], input_id: str) -> bool:
    return any(input_id in user_input_ids_on(message) for message in messages)


def reset_history(session: AgentSession, messages: Sequence[ModelMessage]) -> None:
    session.record(ContextResetEvent())
    run_id = uuid.uuid4().hex
    for sequence, message in enumerate(messages):
        session.record(
            ModelMessageEvent(
                run_id=run_id,
                sequence=sequence,
                message=message,
                context_only=True,
            )
        )


def compact_history(
    session: AgentSession,
    messages: Sequence[ModelMessage],
    step_boundary: bool,
    context_id: str,
    *,
    trigger: Literal["manual", "token_threshold", "tool_request"] = "manual",
) -> None:
    session.record(
        ApplyCompaction(
            messages=list(messages),
            event=CompactionEvent(
                step_boundary=step_boundary,
                trigger=trigger,
                llm_context_id=context_id,
            ),
        )
    )
