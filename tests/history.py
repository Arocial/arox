import uuid
from collections.abc import Sequence
from typing import Literal

from pydantic_ai.messages import ModelMessage

from arox.core.session import (
    AgentSession,
    CompactionEvent,
    ContextResetEvent,
    user_input_ids_on,
)


def record_messages(session: AgentSession, messages: Sequence[ModelMessage]) -> None:
    session.record_model_messages(messages, run_id=uuid.uuid4().hex)


def context_resets(session: AgentSession) -> list[ContextResetEvent]:
    return [entry for entry in session.journal if isinstance(entry, ContextResetEvent)]


def contains_input(messages: Sequence[ModelMessage], input_id: str) -> bool:
    return any(input_id in user_input_ids_on(message) for message in messages)


def reset_history(session: AgentSession, messages: Sequence[ModelMessage]) -> None:
    session.reset_message_history()
    session.record_model_messages(messages, run_id=uuid.uuid4().hex, context_only=True)


def compact_history(
    session: AgentSession,
    messages: Sequence[ModelMessage],
    step_boundary: bool,
    context_id: str,
    *,
    trigger: Literal["manual", "token_threshold", "tool_request"] = "manual",
) -> None:
    session.run_info.llm_context_id = context_id
    session.add_event(
        CompactionEvent(
            agent_name=session.agent_name,
            step_boundary=step_boundary,
            llm_context_id=context_id,
            trigger=trigger,
        )
    )
    reset_history(session, messages)
