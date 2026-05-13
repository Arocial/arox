from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)

_message_adapter = TypeAdapter(ModelMessage)


def _serialize_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return [_message_adapter.dump_python(m, mode="json") for m in messages]


def _deserialize_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return [_message_adapter.validate_python(d) for d in data]


class SessionEvent(BaseModel):
    timestamp: datetime
    event_type: str
    agent_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    agent_name: str
    events: list[SessionEvent] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def add_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            timestamp=datetime.now(UTC),
            event_type=event_type,
            agent_name=self.agent_name,
            data=data or {},
        )
        self.events.append(event)
        return event

    def rebuild_message_history(self) -> list[ModelMessage]:
        """Rebuild message_history from events.

        Walks events in order:
        - agent_step: appends new_messages
        - compaction: resets to compacted_messages
        """
        history: list[ModelMessage] = []
        for event in self.events:
            if event.event_type == "agent_step":
                raw = event.data.get("new_messages", [])
                history.extend(_deserialize_messages(raw))
            elif event.event_type == "compaction":
                raw = event.data.get("compacted_messages", [])
                history = _deserialize_messages(raw)
        return history

    def rebuild_llm_context_id(self) -> str | None:
        """Rebuild llm_context_id from events. Returns None if no context was set."""
        context_id = None
        for event in self.events:
            if event.event_type in ("compaction", "reset"):
                context_id = event.data.get("llm_context_id")
        return context_id

    def user_turn_anchors(self) -> list[int]:
        """Return event indices of ``user_input`` events."""
        return [i for i, ev in enumerate(self.events) if ev.event_type == "user_input"]

    def resolve_user_turn(self, n: int) -> int | None:
        """Resolve "n-th most recent user turn" to an event index."""
        if n < 1:
            return None
        anchors = self.user_turn_anchors()
        if len(anchors) < n:
            return None
        return anchors[-n]

    def truncated_copy(self, event_index: int) -> AgentSession:
        """Return a new :class:`AgentSession` with ``events[0:event_index]``."""
        return AgentSession(
            agent_name=self.agent_name,
            events=[ev.model_copy(deep=True) for ev in self.events[:event_index]],
            extra=dict(self.extra),
        )


class ComposerSession(BaseModel):
    id: str
    composer_name: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[SessionEvent] = Field(default_factory=list)
    agent_sessions: dict[str, AgentSession] = Field(default_factory=dict)
    parent_id: str | None = None
    forked_from: dict[str, int] | None = None

    def fork_at(self, agent_name: str, event_index: int) -> ComposerSession:
        """Create a new ComposerSession truncated at ``event_index`` on ``agent_name``.

        The new session has a fresh id and points back to this one via
        ``parent_id`` / ``forked_from``. The named agent's history is
        truncated; other agents start with fresh (empty) sessions on the
        new branch.
        """
        now = datetime.now(UTC)
        agent_session = self.agent_sessions.get(agent_name)
        if agent_session is None:
            raise ValueError(f"No agent session for '{agent_name}' to fork from")
        new = ComposerSession(
            id=uuid.uuid4().hex[:12],
            composer_name=self.composer_name,
            created_at=now,
            updated_at=now,
            metadata=dict(self.metadata),
            agent_sessions={agent_name: agent_session.truncated_copy(event_index)},
            parent_id=self.id,
            forked_from={agent_name: event_index},
        )
        return new

    def add_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            timestamp=datetime.now(UTC),
            event_type=event_type,
            data=data or {},
        )
        self.events.append(event)
        return event

    def get_agent_session(self, agent_name: str) -> AgentSession:
        if agent_name not in self.agent_sessions:
            self.agent_sessions[agent_name] = AgentSession(agent_name=agent_name)
        return self.agent_sessions[agent_name]

    @staticmethod
    def create(composer_name: str, **metadata: Any) -> ComposerSession:
        now = datetime.now(UTC)
        return ComposerSession(
            id=uuid.uuid4().hex[:12],
            composer_name=composer_name,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )


class SessionStore(Protocol):
    async def list_sessions(self, composer_name: str) -> list[ComposerSession]: ...
    async def load_session(self, session_id: str) -> ComposerSession | None: ...
    async def save_session(self, session: ComposerSession) -> None: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def cleanup(self, max_age_days: int | None = None) -> int: ...


class FileSessionStore:
    def __init__(self, base_dir: Path | None = None, max_age_days: int = 30):
        if base_dir is None:
            base_dir = Path.home() / ".local" / "share" / "arox" / "sessions"
        self.base_dir = base_dir
        self.max_age_days = max_age_days

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _session_meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    async def list_sessions(self, composer_name: str) -> list[ComposerSession]:
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if not meta_path.exists():
                continue
            try:
                raw = json.loads(meta_path.read_text())
                if raw.get("composer_name") == composer_name:
                    session = ComposerSession.model_validate(raw)
                    sessions.append(session)
            except Exception:
                logger.warning(f"Failed to load session from {d}", exc_info=True)

        # Sort sessions by updated_at descending (most recently updated first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def load_session(self, session_id: str) -> ComposerSession | None:
        meta_path = self._session_meta_path(session_id)
        if not meta_path.exists():
            return None

        raw = json.loads(meta_path.read_text())
        session = ComposerSession.model_validate(raw)

        # Load agent sessions
        session_dir = self._session_dir(session_id)
        for state_file in session_dir.glob("agent_*.json"):
            try:
                state_raw = json.loads(state_file.read_text())
                agent_name = state_raw["agent_name"]
                agent_session = AgentSession.model_validate(state_raw)
                session.agent_sessions[agent_name] = agent_session
            except Exception:
                logger.warning(
                    f"Failed to load agent session from {state_file}", exc_info=True
                )

        return session

    async def save_session(self, session: ComposerSession) -> None:
        session.updated_at = datetime.now(UTC)
        session_dir = self._session_dir(session.id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session metadata and events (without agent_sessions inline)
        meta = session.model_dump(mode="json", exclude={"agent_sessions"})
        self._session_meta_path(session.id).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

        # Save each agent session separately
        for agent_name, agent_session in session.agent_sessions.items():
            state_path = session_dir / f"agent_{agent_name}.json"
            state_path.write_text(
                json.dumps(
                    agent_session.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
            )

    async def delete_session(self, session_id: str) -> None:
        import shutil

        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    async def cleanup(self, max_age_days: int | None = None) -> int:
        """Delete sessions older than max_age_days. Returns number of deleted sessions."""
        if not self.base_dir.exists():
            return 0

        max_age = max_age_days if max_age_days is not None else self.max_age_days
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(days=max_age)
        deleted = 0

        for d in list(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if not meta_path.exists():
                continue
            try:
                raw = json.loads(meta_path.read_text())
                updated_at = datetime.fromisoformat(raw.get("updated_at", ""))
                if updated_at < cutoff:
                    import shutil

                    shutil.rmtree(d)
                    deleted += 1
            except Exception:
                logger.warning(f"Failed to check session {d.name}", exc_info=True)

        if deleted:
            logger.info(f"Cleaned up {deleted} expired session(s)")
        return deleted
