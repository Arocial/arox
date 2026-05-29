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

# Metadata key under which a user-turn's session-event id is stored on the
# corresponding ``ModelRequest``. Set in-memory during a step and re-derived
# from ``user_input`` events when a session is restored.
USER_INPUT_ID_KEY = "user_input_id"


def _serialize_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return [_message_adapter.dump_python(m, mode="json") for m in messages]


def _deserialize_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return [_message_adapter.validate_python(d) for d in data]


class SessionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime
    event_type: str
    agent_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forked_from: dict[str, int] | None = None
    owner_id: str | None = None
    owner_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[SessionEvent] = Field(default_factory=list)

    def add_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        id: str | None = None,
    ) -> SessionEvent:
        kwargs: dict[str, Any] = {
            "timestamp": datetime.now(UTC),
            "event_type": event_type,
            "data": data or {},
        }
        if id is not None:
            kwargs["id"] = id
        event = SessionEvent(**kwargs)
        self.events.append(event)
        return event

    def index_of_event(self, event_id: str) -> int | None:
        for i, ev in enumerate(self.events):
            if ev.id == event_id:
                return i
        return None


def derive_child_session_id(owner_session_id: str, agent_name: str) -> str:
    """Derive a stable session id for an agent nested under ``owner_session_id``.

    Deterministic in ``(owner_session_id, agent_name)`` so that a subagent's
    session can be located on resume without persisting a separate index: the
    owner re-derives the same id from its own (possibly forked) session id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{owner_session_id}/{agent_name}"))


_SESSION_TYPES: dict[str, type[Session]] = {}


def register_session_type(cls: type[Session]) -> None:
    """Register a Session subclass so the store can deserialize it by session_type."""
    type_name = cls.model_fields["session_type"].default
    _SESSION_TYPES[type_name] = cls


class SessionStore(Protocol):
    async def list_sessions(self, session_type: str = "agent") -> list[Session]: ...
    async def load_session(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> Session | None: ...
    async def save_session(self, session: Session) -> None: ...
    async def delete_session(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> None: ...
    async def cleanup(self, max_age_days: int | None = None) -> int: ...


class FileSessionStore:
    def __init__(self, base_dir: Path | None = None, max_age_days: int = 30):
        if base_dir is None:
            base_dir = Path.home() / ".local" / "share" / "arox" / "sessions"
        self.base_dir = base_dir
        self.max_age_days = max_age_days

    def _session_dir(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> Path:
        path = self.base_dir
        if owner_path:
            for owner in owner_path:
                path = path / owner
        return path / session_id

    def _session_meta_path(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> Path:
        return self._session_dir(session_id, owner_path) / "session.json"

    async def list_sessions(self, session_type: str = "agent") -> list[Session]:
        if not self.base_dir.exists():
            return []
        sessions: list[Session] = []
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if not meta_path.exists():
                continue
            try:
                raw = json.loads(meta_path.read_text())
                if raw.get("session_type") != session_type:
                    continue
                model = _SESSION_TYPES.get(session_type, Session)
                sessions.append(model.model_validate(raw))
            except Exception:
                logger.warning(f"Failed to load session from {d}", exc_info=True)

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def load_session(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> Session | None:
        meta_path = self._session_meta_path(session_id, owner_path)
        if not meta_path.exists():
            return None

        raw = json.loads(meta_path.read_text())
        session_type = raw.get("session_type")
        model = _SESSION_TYPES.get(session_type, Session)
        return model.model_validate(raw)

    async def save_session(self, session: Session) -> None:
        session.updated_at = datetime.now(UTC)
        session_dir = self._session_dir(session.id, session.owner_path)
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = session.model_dump(mode="json")
        self._session_meta_path(session.id, session.owner_path).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    async def delete_session(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> None:
        import shutil

        session_dir = self._session_dir(session_id, owner_path)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    async def cleanup(self, max_age_days: int | None = None) -> int:
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
