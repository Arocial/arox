from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Union

from pydantic import BaseModel, Field, model_validator
from pydantic_ai.messages import (
    ModelMessage,
)

from arox.core.config import AgentConfig

logger = logging.getLogger(__name__)


# Metadata key under which a user-turn's session-event id is stored on the
# corresponding ``ModelRequest``. Set in-memory during a step and re-derived
# from ``user_input`` events when a session is restored.
USER_INPUT_ID_KEY = "user_input_id"


class SessionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    agent_name: str = ""

    @model_validator(mode="before")
    @classmethod
    def _migrate_data(cls, data: Any) -> Any:
        if isinstance(data, dict) and "data" in data:
            extra = data.pop("data")
            if isinstance(extra, dict):
                data.update(extra)
        return data


class ResetEvent(SessionEvent):
    event_type: Literal["reset"] = "reset"
    llm_context_id: str = ""


class StepEvent(SessionEvent):
    event_type: Literal["agent_step"] = "agent_step"
    new_messages: list[ModelMessage] = Field(default_factory=list)
    request_tokens: int | None = None
    response_tokens: int | None = None


class CommandEvent(SessionEvent):
    event_type: Literal["command"] = "command"
    command: str = ""
    arg: str | None = None


class UserInputEvent(SessionEvent):
    event_type: Literal["user_input"] = "user_input"
    text: str = ""


class ErrorEvent(SessionEvent):
    event_type: Literal["error"] = "error"
    error: str = ""


class SubagentCallEvent(SessionEvent):
    event_type: Literal["subagent_call"] = "subagent_call"
    subagent: str = ""
    task: str = ""


class SubagentCreatedEvent(SessionEvent):
    event_type: Literal["subagent_created"] = "subagent_created"
    subagent: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class SubagentDeletedEvent(SessionEvent):
    event_type: Literal["subagent_deleted"] = "subagent_deleted"
    subagent: str = ""
    session_id: str | None = None


class CompactionEvent(SessionEvent):
    event_type: Literal["compaction"] = "compaction"
    compacted_messages: list[ModelMessage] = Field(default_factory=list)
    step_boundary: bool = False
    llm_context_id: str = ""


AnySessionEvent = Annotated[
    Union[
        ResetEvent,
        StepEvent,
        CommandEvent,
        UserInputEvent,
        ErrorEvent,
        SubagentCallEvent,
        SubagentCreatedEvent,
        SubagentDeletedEvent,
        CompactionEvent,
    ],
    Field(discriminator="event_type"),
]


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forked_from: dict[str, int] | None = None
    owner_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[AnySessionEvent] = Field(default_factory=list)
    # Session ids owned by this one (its subsessions).
    children: list[str] = Field(default_factory=list)

    manager: Any = Field(default=None, exclude=True, repr=False)
    owner: Any = Field(default=None, exclude=True, repr=False)

    async def save(self) -> None:
        if self.manager:
            # Explicit save bypasses debounce; remove from dirty set to avoid redundant writes.
            self.manager._dirty_sessions.pop(self.id, None)
            await self.manager.save_session(self)

    def add_event(
        self,
        event: AnySessionEvent,
    ) -> AnySessionEvent:
        self.events.append(event)
        if self.manager:
            self.manager.notify_dirty(self)
        return event

    async def build_instance(self, session_manager: SessionManager, **kwargs) -> Any:
        raise NotImplementedError

    def index_of_event(self, event_id: str) -> int | None:
        for i, ev in enumerate(self.events):
            if ev.id == event_id:
                return i
        return None


class AgentSession(Session):
    session_type: str = "agent"
    agent_name: str
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    agent_source: Literal["static", "dynamic"] = "dynamic"
    workspace: str | None = None
    llm_context_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_initial(
        cls,
        parsed_config: Any,
        workspace: str | Path | None = None,
    ) -> AgentSession:
        """Create a fresh session for the main agent from config."""
        agent_name = parsed_config.app.main_agent
        agent_config = parsed_config.agent.get(agent_name) or AgentConfig()
        ws = str(Path(workspace).absolute()) if workspace else str(Path.cwd())
        return cls(
            id=str(uuid.uuid4()),
            agent_name=agent_name,
            agent_config=agent_config.model_copy(deep=True),
            agent_source="static",
            workspace=ws,
            llm_context_id=str(uuid.uuid4()),
        )

    async def build_instance(self, session_manager: SessionManager, **kwargs) -> Any:
        from arox.utils import import_class

        agent_cls = import_class(self.agent_config.type, group="arox.agents")

        return agent_cls(
            session=self,
            **kwargs,
        )

    def rebuild_message_history(self) -> list[ModelMessage]:
        history: list[ModelMessage] = []
        for event in self.events:
            if isinstance(event, StepEvent):
                history.extend(event.new_messages)
            elif isinstance(event, CompactionEvent) and event.step_boundary:
                history = list(event.compacted_messages)
            elif isinstance(event, ResetEvent) or (
                isinstance(event, CompactionEvent) and not event.step_boundary
            ):
                history = []
        return history

    def rebuild_llm_context_id(self) -> str | None:
        context_id = self.llm_context_id
        for event in self.events:
            if isinstance(event, (CompactionEvent, ResetEvent)):
                context_id = event.llm_context_id
        return context_id

    def record_reset(self, llm_context_id: str) -> None:
        self.llm_context_id = llm_context_id
        self.add_event(
            ResetEvent(llm_context_id=llm_context_id, agent_name=self.agent_name)
        )
        self.metadata.pop("last_user_messages", None)

    def record_step(
        self,
        new_messages: Sequence[ModelMessage],
        request_tokens: int | None = None,
        response_tokens: int | None = None,
    ) -> None:
        self.add_event(
            StepEvent(
                new_messages=list(new_messages),
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                agent_name=self.agent_name,
            )
        )

    def record_command(self, command: str, arg: str | None) -> None:
        self.add_event(
            CommandEvent(command=command, arg=arg, agent_name=self.agent_name)
        )

    def record_user_input(self, text: str, input_id: str) -> None:
        self.add_event(
            UserInputEvent(id=input_id, text=text, agent_name=self.agent_name)
        )
        last_user_messages = self.metadata.get("last_user_messages", [])
        last_user_messages.append(text)
        self.metadata["last_user_messages"] = last_user_messages[-2:]

    def record_error(self, error: Exception) -> None:
        self.add_event(
            ErrorEvent(
                error=f"{type(error).__name__}: {error!s}", agent_name=self.agent_name
            )
        )

    def record_subagent_call(self, subagent_name: str, task: str) -> None:
        self.add_event(
            SubagentCallEvent(
                subagent=subagent_name, task=task, agent_name=self.agent_name
            )
        )

    def record_subagent_created(
        self, subagent_name: str, config_data: dict[str, Any]
    ) -> None:
        self.add_event(
            SubagentCreatedEvent(
                subagent=subagent_name, config=config_data, agent_name=self.agent_name
            )
        )

    def record_subagent_deleted(
        self, subagent_name: str, session_id: str | None
    ) -> None:
        self.add_event(
            SubagentDeletedEvent(
                subagent=subagent_name,
                session_id=session_id,
                agent_name=self.agent_name,
            )
        )

    def record_compaction(
        self,
        compacted_messages: list[ModelMessage],
        step_boundary: bool,
        llm_context_id: str,
    ) -> None:
        self.llm_context_id = llm_context_id
        self.add_event(
            CompactionEvent(
                step_boundary=step_boundary,
                compacted_messages=compacted_messages,
                llm_context_id=llm_context_id,
                agent_name=self.agent_name,
            )
        )

    async def fork_at(self, event_id: str | None) -> "AgentSession":
        """Branch a new session off this one

        With ``event_id`` the new session is a truncated copy holding the events
        up to (but excluding) that anchor, tagged via ``forked_from``. With
        ``event_id`` set to ``None`` it starts empty and unforked.
        """
        owner = self.owner

        forked_from: dict[str, int] | None
        if event_id is None:
            events = []
            forked_from = None
        else:
            event_index = self.index_of_event(event_id)
            if event_index is None:
                raise ValueError(f"event {event_id} not found")
            events = [ev.model_copy(deep=True) for ev in self.events[:event_index]]
            forked_from = {self.agent_name: event_index}
        new_id = str(uuid.uuid4())

        new_session = AgentSession(
            id=new_id,
            agent_name=self.agent_name,
            agent_config=self.agent_config.model_copy(deep=True),
            agent_source=self.agent_source,
            workspace=self.workspace,
            llm_context_id=self.llm_context_id,
            owner_path=[*owner.owner_path, owner.id] if owner else [],
            events=events,
            extra=dict(self.extra),
            forked_from=forked_from,
        )
        new_session.manager = self.manager
        new_session.owner = owner
        if owner:
            owner.children.append(new_session.id)

        for child_id in self.children:
            try:
                if self.manager:
                    sub_session = await self.manager.load_session(child_id, self)
                else:
                    logger.warning(
                        f"No session manager to load child session {child_id}"
                    )
                    continue
            except Exception:
                logger.warning(
                    f"Failed to load child session {child_id}", exc_info=True
                )
                continue

            if sub_session:
                forked_child = await sub_session.fork_at(None, new_session)
                await forked_child.save()

        return new_session


class SessionManager:
    def __init__(self, session_store: "SessionStore"):
        self.session_store = session_store
        self._session_types: dict[str, type[Session]] = {}
        self.session_store.set_session_types(self._session_types)

        self._dirty_sessions: dict[str, Session] = {}
        self._save_event: asyncio.Event = asyncio.Event()
        self._save_task: asyncio.Task | None = None

    async def __aenter__(self):
        self._save_task = asyncio.get_running_loop().create_task(self._save_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._save_task is not None:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None
        # Final flush for anything queued after the last loop iteration.
        await self._flush_dirty_sessions()

    def notify_dirty(self, session: Session) -> None:
        self._dirty_sessions[session.id] = session
        self._save_event.set()

    async def _save_loop(self) -> None:
        try:
            while True:
                await self._save_event.wait()
                self._save_event.clear()

                # Debounce: let rapid-fire events coalesce.
                await asyncio.sleep(0.1)
                await self._flush_dirty_sessions()
        except asyncio.CancelledError:
            await asyncio.shield(self._flush_dirty_sessions())
            raise

    async def _flush_dirty_sessions(self) -> None:
        if not self._dirty_sessions:
            return
        to_save = list(self._dirty_sessions.values())
        self._dirty_sessions.clear()
        for session in to_save:
            try:
                await self.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session {session.id}: {e}", exc_info=True)
                # Re-queue for retry on the next cycle.
                self._dirty_sessions.setdefault(session.id, session)

    def register_session_type(self, cls: type[Session]) -> None:
        """Register a Session subclass so the store can deserialize it by session_type."""
        type_name = cls.model_fields["session_type"].default
        self._session_types[type_name] = cls

    async def save_session(self, session: Session) -> None:
        await self.session_store.save_session(session)

    async def load_session(self, session_id, owner: Session | None) -> Session | None:
        owner_path = [*owner.owner_path, owner.id] if owner else []
        session = await self.session_store.load_session(session_id, owner_path)
        if session:
            session.owner = owner
        return session

    async def build_from_session(
        self, session_id: str, owner: Session | None = None, **kwargs
    ) -> Any:

        session = await self.load_session(session_id, owner)
        if not session:
            return None
        session.manager = self
        return await session.build_instance(self, **kwargs)


class SessionStore(Protocol):
    def set_session_types(self, session_types: dict[str, type[Session]]) -> None: ...
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
        self._session_types: dict[str, type[Session]] = {}

    def set_session_types(self, session_types: dict[str, type[Session]]) -> None:
        self._session_types = session_types

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
                model = self._session_types.get(session_type, Session)
                sessions.append(model.model_validate(raw))
            except Exception:
                logger.warning(f"Failed to load session from {d}", exc_info=True)

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def load_session(
        self, session_id: str, owner_path: list[str] | None = None
    ) -> Session | None:
        owner_path = owner_path or []
        meta_path = self._session_meta_path(session_id, owner_path)
        if not meta_path.exists():
            return None

        raw = json.loads(meta_path.read_text())
        session_type = raw.get("session_type")
        model = self._session_types.get(session_type, Session)
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
