from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, Union

if TYPE_CHECKING:
    from arox.core.config import ConfigLoader
    from arox.core.io import AbstractIOAdapter
    from arox.core.llm_base.agent import LLMBaseAgent

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    ModelMessage,
)

from arox.core.types import UserInput

logger = logging.getLogger(__name__)


class SessionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    agent_name: str = ""


class ResetEvent(SessionEvent):
    event_type: Literal["reset"] = "reset"
    llm_context_id: str = ""


class StepEvent(SessionEvent):
    event_type: Literal["agent_step"] = "agent_step"
    new_messages: list[ModelMessage] = Field(default_factory=list)


class CommandEvent(SessionEvent):
    event_type: Literal["command"] = "command"
    command: str = ""
    arg: str | None = None


class UserInputEvent(SessionEvent):
    event_type: Literal["user_input"] = "user_input"
    user_input: UserInput


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
    session_path: list[str] | None = None


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


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: list[str] = Field(default_factory=lambda: [str(uuid.uuid4())])
    session_type: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forked_from: tuple[list[str], str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[AnySessionEvent] = Field(default_factory=list)
    # Session ids owned by this one (its subsessions).
    children: list[str] = Field(default_factory=list)
    initialized: bool = False

    manager: Any = Field(default=None, exclude=True, repr=False)
    owner: Any = Field(default=None, exclude=True, repr=False)

    @property
    def id(self) -> str:
        return self.path[-1]

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

    def index_of_event(self, event_id: str) -> int | None:
        for i, ev in enumerate(self.events):
            if ev.id == event_id:
                return i
        return None


class AgentRunInfo(BaseModel):
    context_tokens: int = 0
    total_tokens: int = 0
    llm_context_id: str | None = None
    run_id: str | None = None


class AgentSession(Session):
    session_type: str = "agent"
    agent_name: str
    agent_type: str = "chat"
    agent_source: Literal["static", "dynamic"] = "dynamic"
    workspace: str | None = None
    task_name: str | None = None
    target: str | None = None
    initial_message: str | None = None
    last_message: str | None = None
    last_result: str | None = None
    last_error: str | None = None
    run_info: AgentRunInfo = Field(default_factory=AgentRunInfo)
    extra: dict[str, Any] = Field(default_factory=dict)

    runtime: Any = Field(default=None, exclude=True, repr=False)
    running_task: asyncio.Task[Any] | None = Field(
        default=None, exclude=True, repr=False
    )

    @property
    def task_id(self) -> str:
        return self.id

    @property
    def agent(self) -> Any:
        return self.runtime

    @property
    def has_runtime(self) -> bool:
        return self.runtime is not None

    @property
    def result(self) -> str | None:
        return self.last_result

    @property
    def error(self) -> str | None:
        return self.last_error

    @property
    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE

    @property
    def is_running(self) -> bool:
        return (
            self.is_active
            and self.running_task is not None
            and not self.running_task.done()
        )

    def record_result(self, result: str | None) -> None:
        self.last_result = result
        self.last_error = None

    def record_interrupted(self, message: str = "Task interrupted.") -> None:
        self.last_error = message

    def close_session(self) -> None:
        self.status = SessionStatus.CLOSED
        self.runtime = None

    def create_child_session(
        self,
        agent_name: str,
        *,
        agent_type: str = "chat",
        agent_source: Literal["static", "dynamic"] = "static",
        workspace: Path | str | None = None,
        task_name: str | None = None,
        target: str | None = None,
        initial_message: str | None = None,
        last_message: str | None = None,
        status: SessionStatus = SessionStatus.ACTIVE,
    ) -> AgentSession:
        child_workspace = (
            str(Path(workspace).absolute()) if workspace is not None else self.workspace
        )
        child = AgentSession(
            path=[*self.path, str(uuid.uuid4())],
            agent_name=agent_name,
            agent_type=agent_type,
            agent_source=agent_source,
            workspace=child_workspace,
            task_name=task_name,
            target=target,
            initial_message=initial_message,
            last_message=last_message,
            status=status,
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
            owner=self,
            manager=self.manager,
        )
        self.children.append(child.id)
        return child

    def create_agent(
        self,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
    ) -> LLMBaseAgent:
        from arox import utils

        agent_config = config_loader.current_config.agent.get(self.agent_name)
        if not agent_config:
            raise ValueError(f"Agent config for '{self.agent_name}' not found")

        agent_type = agent_config.type or self.agent_type
        try:
            agent_cls = utils.import_class(agent_type, group="arox.agents")
        except ValueError:
            raise ValueError(
                f"Unknown agent type: {agent_type} for agent {self.agent_name}"
            )

        return agent_cls(
            parent_config_loader=config_loader,
            io_adapter=io_adapter,
            session=self,
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
        context_id = self.run_info.llm_context_id
        for event in self.events:
            if isinstance(event, (CompactionEvent, ResetEvent)):
                context_id = event.llm_context_id
        return context_id

    def record_reset(self, llm_context_id: str) -> None:
        self.run_info.llm_context_id = llm_context_id
        self.add_event(
            ResetEvent(llm_context_id=llm_context_id, agent_name=self.agent_name)
        )
        self.metadata.pop("last_user_messages", None)

    def record_step(
        self,
        new_messages: Sequence[ModelMessage],
    ) -> None:
        self.add_event(
            StepEvent(
                new_messages=list(new_messages),
                agent_name=self.agent_name,
            )
        )

    def record_command(self, command: str, arg: str | None) -> None:
        self.add_event(
            CommandEvent(command=command, arg=arg, agent_name=self.agent_name)
        )

    def record_user_input(self, user_input: UserInput) -> None:
        input_id = user_input.server_message_id

        self.add_event(
            UserInputEvent(
                id=input_id, user_input=user_input, agent_name=self.agent_name
            )
        )
        last_user_messages = self.metadata.get("last_user_messages", [])
        text = user_input.text_content
        if text:
            last_user_messages.append(text)
        self.metadata["last_user_messages"] = last_user_messages[-2:]

    def record_error(self, error: Exception | str) -> None:
        err_msg = (
            str(error)
            if isinstance(error, str)
            else f"{type(error).__name__}: {error!s}"
        )
        self.last_error = err_msg
        self.add_event(ErrorEvent(error=err_msg, agent_name=self.agent_name))

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
        self, subagent_name: str, session_path: list[str] | None
    ) -> None:
        self.add_event(
            SubagentDeletedEvent(
                subagent=subagent_name,
                session_path=session_path,
                agent_name=self.agent_name,
            )
        )

    def record_compaction(
        self,
        compacted_messages: list[ModelMessage],
        step_boundary: bool,
        llm_context_id: str,
    ) -> None:
        self.run_info.llm_context_id = llm_context_id
        self.add_event(
            CompactionEvent(
                step_boundary=step_boundary,
                compacted_messages=compacted_messages,
                llm_context_id=llm_context_id,
                agent_name=self.agent_name,
            )
        )

    async def fork_at(
        self, event_id: str | None, new_owner: Session | None = None
    ) -> "AgentSession":
        """Branch a new session off this one

        With ``event_id`` the new session is a truncated copy holding the events
        up to (but excluding) that anchor, tagged via ``forked_from``. With
        ``event_id`` set to ``None`` it starts empty and unforked.
        """
        owner = new_owner or self.owner

        forked_from: tuple[list[str], str] | None
        if event_id is None:
            events = []
            forked_from = None
        else:
            event_index = self.index_of_event(event_id)
            if event_index is None:
                raise ValueError(f"event {event_id} not found")
            events = [ev.model_copy(deep=True) for ev in self.events[:event_index]]
            forked_from = (self.path, event_id)

        manager_ref = self.manager
        owner_ref = self.owner
        self.manager = None
        self.owner = None
        try:
            new_session = self.model_copy(
                deep=True,
                update={
                    "path": [*owner.path, str(uuid.uuid4())]
                    if owner
                    else [str(uuid.uuid4())],
                    "events": events,
                    "forked_from": forked_from,
                    "children": [],
                    "status": SessionStatus.ACTIVE,
                    "last_result": None,
                    "last_error": None,
                    "manager": manager_ref,
                    "owner": owner,
                },
            )
        finally:
            self.manager = manager_ref
            self.owner = owner_ref
        new_session.run_info.llm_context_id = str(uuid.uuid4())

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

    async def load_session(
        self, session_id: str, owner: Session | None
    ) -> Session | None:
        path = [*owner.path, session_id] if owner else [session_id]
        session = await self.session_store.load_session(path)
        if session:
            session.owner = owner
        return session


class SessionStore(Protocol):
    def set_session_types(self, session_types: dict[str, type[Session]]) -> None: ...
    def session_dir(self, path: list[str]) -> Path: ...
    async def list_sessions(self, session_type: str = "agent") -> list[Session]: ...
    async def load_session(self, path: list[str]) -> Session | None: ...
    async def save_session(self, session: Session) -> None: ...
    async def delete_session(self, path: list[str]) -> None: ...
    async def cleanup(self, max_age_days: int | None = None) -> int: ...


class FileSessionStore:
    def __init__(self, namespace: str = "default", max_age_days: int = 30):
        from platformdirs import user_data_dir

        self.base_dir = Path(user_data_dir("arox")) / "sessions" / namespace
        self.max_age_days = max_age_days
        self._session_types: dict[str, type[Session]] = {}

    def set_session_types(self, session_types: dict[str, type[Session]]) -> None:
        self._session_types = session_types

    def session_dir(self, path: list[str]) -> Path:
        return self.base_dir.joinpath(*path)

    def _session_meta_path(self, path: list[str]) -> Path:
        return self.session_dir(path) / "session.json"

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

    async def load_session(self, path: list[str]) -> Session | None:
        meta_path = self._session_meta_path(path)
        if not meta_path.exists():
            return None

        raw = json.loads(meta_path.read_text())
        session_type = raw.get("session_type")
        model = self._session_types.get(session_type, Session)
        return model.model_validate(raw)

    async def save_session(self, session: Session) -> None:
        session.updated_at = datetime.now(UTC)
        session_dir = self.session_dir(session.path)
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = session.model_dump(mode="json")
        self._session_meta_path(session.path).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    async def delete_session(self, path: list[str]) -> None:
        import shutil

        session_dir = self.session_dir(path)
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
