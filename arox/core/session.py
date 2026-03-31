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

    def rebuild_message_history(
        self, example_messages: Sequence[ModelMessage]
    ) -> list[ModelMessage]:
        """Rebuild message_history from events.

        Walks events in order:
        - agent_step: appends new_messages
        - compaction: resets to example_messages + compacted_messages
        """
        history = list(example_messages)
        for event in self.events:
            if event.event_type == "agent_step":
                raw = event.data.get("new_messages", [])
                history.extend(_deserialize_messages(raw))
            elif event.event_type == "compaction":
                raw = event.data.get("compacted_messages", [])
                history = list(example_messages) + _deserialize_messages(raw)
        return history


class AppSession(BaseModel):
    id: str
    composer_name: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[SessionEvent] = Field(default_factory=list)
    agent_sessions: dict[str, AgentSession] = Field(default_factory=dict)

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
    def create(composer_name: str, **metadata: Any) -> AppSession:
        now = datetime.now(UTC)
        return AppSession(
            id=uuid.uuid4().hex[:12],
            composer_name=composer_name,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )


class SessionStore(Protocol):
    async def list_sessions(self, composer_name: str) -> list[AppSession]: ...
    async def load_session(self, session_id: str) -> AppSession | None: ...
    async def save_session(self, session: AppSession) -> None: ...
    async def delete_session(self, session_id: str) -> None: ...


class FileSessionStore:
    def __init__(self, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".local" / "share" / "arox" / "sessions"
        self.base_dir = base_dir

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _session_meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    async def list_sessions(self, composer_name: str) -> list[AppSession]:
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if not meta_path.exists():
                continue
            try:
                raw = json.loads(meta_path.read_text())
                if raw.get("composer_name") == composer_name:
                    session = AppSession.model_validate(raw)
                    sessions.append(session)
            except Exception:
                logger.warning(f"Failed to load session from {d}", exc_info=True)
        return sessions

    async def load_session(self, session_id: str) -> AppSession | None:
        meta_path = self._session_meta_path(session_id)
        if not meta_path.exists():
            return None

        raw = json.loads(meta_path.read_text())
        session = AppSession.model_validate(raw)

        # Load agent sessions
        session_dir = self._session_dir(session_id)
        for state_file in session_dir.glob("agent_*.json"):
            try:
                state_raw = json.loads(state_file.read_text())
                agent_name = state_raw["agent_name"]
                session.agent_sessions[agent_name] = AgentSession.model_validate(
                    state_raw
                )
            except Exception:
                logger.warning(
                    f"Failed to load agent session from {state_file}", exc_info=True
                )

        return session

    async def save_session(self, session: AppSession) -> None:
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
