from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_ai.messages import ModelMessage, ModelRequest, TextContent, UserPromptPart

from arox.core.types import USER_INPUT_ID_KEY, UserInput

logger = logging.getLogger(__name__)


class SessionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    agent_name: str = ""


class StepEvent(SessionEvent):
    event_type: Literal["agent_step"] = "agent_step"


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
    step_boundary: bool = False
    llm_context_id: str = ""


AnySessionEvent = Annotated[
    Union[
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: list[str] = Field(default_factory=lambda: [str(uuid.uuid4())])
    session_type: str
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


class MessageHistorySegment(BaseModel):
    messages: list[ModelMessage] = Field(default_factory=list)

    @staticmethod
    def _user_input_ids_on(message: ModelMessage) -> list[str]:
        if not isinstance(message, ModelRequest):
            return []

        input_ids: list[str] = []
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            for item in part.content:
                if not isinstance(item, TextContent) or not item.metadata:
                    continue
                input_id = item.metadata.get(USER_INPUT_ID_KEY)
                if isinstance(input_id, str):
                    input_ids.append(input_id)
        return input_ids

    def contains_user_input(self, input_id: str) -> bool:
        return any(
            input_id in self._user_input_ids_on(message) for message in self.messages
        )

    def prefix_before_user_input(self, input_id: str) -> list[ModelMessage]:
        for index, message in enumerate(self.messages):
            if input_id in self._user_input_ids_on(message):
                return list(self.messages[:index])
        raise KeyError(input_id)

    def copy_before_user_input(
        self, input_id: str | None = None
    ) -> "MessageHistorySegment":
        messages = (
            self.prefix_before_user_input(input_id)
            if input_id is not None
            else self.messages
        )
        return MessageHistorySegment(messages=copy.deepcopy(messages))


class AgentSession(Session):
    session_type: str = "agent"
    agent_name: str
    agent_source: Literal["static", "dynamic"] = "dynamic"
    workspace: str | None = None
    task_name: str | None = None
    target: str | None = None
    initial_message: str | None = None
    last_message: str | None = None
    result: str | None = None
    error: str | None = None
    run_info: AgentRunInfo = Field(default_factory=AgentRunInfo)
    archived_message_histories: list[MessageHistorySegment] = Field(
        default_factory=list
    )
    message_history: MessageHistorySegment = Field(
        default_factory=MessageHistorySegment
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    runner: Any = Field(default=None, exclude=True, repr=False)
    _runner_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def task_id(self) -> str:
        return self.id

    @property
    def runtime(self) -> Any:
        return self.runner.runtime if self.runner is not None else None

    @property
    def is_active(self) -> bool:
        return self.runner is not None

    def create_child_session(
        self,
        agent_name: str,
        *,
        agent_source: Literal["static", "dynamic"] = "static",
        workspace: Path | str | None = None,
        task_name: str | None = None,
        target: str | None = None,
        initial_message: str | None = None,
        last_message: str | None = None,
    ) -> AgentSession:
        child_workspace = (
            str(Path(workspace).absolute()) if workspace is not None else self.workspace
        )
        child = AgentSession(
            path=[*self.path, str(uuid.uuid4())],
            agent_name=agent_name,
            agent_source=agent_source,
            workspace=child_workspace,
            task_name=task_name,
            target=target,
            initial_message=initial_message,
            last_message=last_message,
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
            owner=self,
            manager=self.manager,
        )
        self.children.append(child.id)
        if self.manager:
            self.manager._track(self, self.owner)
            self.manager._track(child, self)
        return child

    def replace_message_history(self, messages: Sequence[ModelMessage]) -> None:
        self.message_history.messages = list(messages)

    def _start_message_history(
        self,
        messages: Sequence[ModelMessage],
        *,
        previous_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        history_to_archive = (
            self.message_history
            if previous_messages is None
            else MessageHistorySegment(messages=list(previous_messages))
        )
        if history_to_archive.messages:
            self.archived_message_histories.append(history_to_archive)
        self.message_history = MessageHistorySegment(messages=list(messages))

    def record_step(
        self,
        message_history: Sequence[ModelMessage],
    ) -> None:
        self.add_event(StepEvent(agent_name=self.agent_name))
        self.replace_message_history(message_history)

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

    def record_turn_error(self, error: Exception | str) -> None:
        error_message = self.record_error_event(error)
        self.error = error_message

    def record_error_event(self, error: Exception | str) -> str:
        """Record an error event without changing task scheduling state."""
        err_msg = (
            str(error)
            if isinstance(error, str)
            else f"{type(error).__name__}: {error!s}"
        )
        self.add_event(ErrorEvent(error=err_msg, agent_name=self.agent_name))
        return err_msg

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
        compacted_messages: Sequence[ModelMessage],
        step_boundary: bool,
        llm_context_id: str,
        *,
        previous_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        self.run_info.llm_context_id = llm_context_id
        self.add_event(
            CompactionEvent(
                step_boundary=step_boundary,
                llm_context_id=llm_context_id,
                agent_name=self.agent_name,
            )
        )
        self._start_message_history(
            compacted_messages, previous_messages=previous_messages
        )

    def _fork_message_histories(
        self, input_id: str
    ) -> tuple[list[MessageHistorySegment], MessageHistorySegment]:
        histories = [*self.archived_message_histories, self.message_history]
        for index, history in enumerate(histories):
            if not history.contains_user_input(input_id):
                continue

            archived = [
                previous.copy_before_user_input() for previous in histories[:index]
            ]
            active = history.copy_before_user_input(input_id)
            return archived, active

        raise ValueError(f"user input {input_id} has no message history")

    async def fork_at(
        self, event_id: str | None, new_owner: Session | None = None
    ) -> "AgentSession":
        """Branch a new session before a user-input request."""
        owner = new_owner or self.owner

        forked_from: tuple[list[str], str] | None
        if event_id is None:
            events = []
            archived_message_histories = []
            message_history = MessageHistorySegment()
            forked_from = None
        else:
            event_index = self.index_of_event(event_id)
            if event_index is None:
                raise ValueError(f"event {event_id} not found")
            if not isinstance(self.events[event_index], UserInputEvent):
                raise ValueError(f"event {event_id} is not a user input")
            events = [ev.model_copy(deep=True) for ev in self.events[:event_index]]
            archived_message_histories, message_history = self._fork_message_histories(
                event_id
            )
            forked_from = (self.path, event_id)

        manager_ref = self.manager
        copy_source = self.model_copy(
            update={"manager": None, "owner": None, "runner": None}
        )
        new_session = copy_source.model_copy(
            deep=True,
            update={
                "path": [*owner.path, str(uuid.uuid4())]
                if owner
                else [str(uuid.uuid4())],
                "events": events,
                "archived_message_histories": archived_message_histories,
                "message_history": message_history,
                "forked_from": forked_from,
                "children": [],
                "task_name": None,
                "target": None,
                "initial_message": None,
                "last_message": None,
                "result": None,
                "error": None,
                "manager": manager_ref,
                "owner": owner,
                "runner": None,
            },
        )
        new_session.run_info.llm_context_id = str(uuid.uuid4())

        if owner:
            owner.children.append(new_session.id)

        child_sessions = await self.manager.children_of(self) if self.manager else []
        if not self.manager and self.children:
            logger.warning("No session manager to load child sessions")
        for sub_session in child_sessions:
            try:
                forked_child = await sub_session.fork_at(None, new_session)
                await forked_child.save()
            except Exception:
                logger.warning(
                    "Failed to fork child session %s", sub_session.id, exc_info=True
                )

        return new_session


class SessionManager:
    def __init__(self, session_store: "SessionStore"):
        self.session_store = session_store
        self._session_types: dict[str, type[Session]] = {}
        self.session_store.set_session_types(self._session_types)

        self._dirty_sessions: dict[str, Session] = {}
        self._sessions: dict[tuple[str, ...], Session] = {}
        self._save_event: asyncio.Event = asyncio.Event()
        self._save_task: asyncio.Task | None = None

    async def __aenter__(self):
        self._save_task = asyncio.get_running_loop().create_task(self._save_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_all()
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

    def _track(self, session: Session, owner: Session | None = None) -> Session:
        session.manager = self
        session.owner = owner
        self._sessions[tuple(session.path)] = session
        return session

    def _forget_tree(self, path: list[str]) -> None:
        prefix = tuple(path)
        for cached_path in list(self._sessions):
            if cached_path[: len(prefix)] == prefix:
                self._sessions.pop(cached_path, None)

    async def stop_all(self) -> None:
        """Stop every live root tree managed by this instance."""
        roots = [
            session
            for session in self._sessions.values()
            if isinstance(session, AgentSession)
            and session.is_active
            and len(session.path) == 1
        ]
        await asyncio.gather(
            *(self.stop_tree(session) for session in roots),
            return_exceptions=True,
        )

    async def save_session(self, session: Session) -> None:
        self._track(session, session.owner)
        await self.session_store.save_session(session)

    async def resolve(
        self, session_id: str, owner: Session | None = None
    ) -> Session | None:
        """Resolve a root or direct child, preferring its live instance."""
        path = [*owner.path, session_id] if owner else [session_id]
        cached = self._sessions.get(tuple(path))
        if cached is not None:
            return self._track(cached, owner)

        session = await self.session_store.load_session(path)
        if session:
            self._track(session, owner)
        return session

    async def list_roots(self, session_type: str = "agent") -> list[Session]:
        roots = await self.session_store.list_sessions(session_type)
        resolved = []
        for stored in roots:
            root = self._sessions.get(tuple(stored.path), stored)
            self._track(root)
            resolved.append(root)
        return resolved

    async def children_of(self, parent: Session) -> list[Session]:
        children = []
        for child_id in parent.children:
            child = await self.resolve(child_id, parent)
            if child is not None:
                children.append(child)
        return children

    async def walk(self, root: Session) -> list[Session]:
        sessions = [root]
        for child in await self.children_of(root):
            sessions.extend(await self.walk(child))
        return sessions

    async def find(self, root: Session, session_id: str) -> Session | None:
        if root.id == session_id:
            return root
        for child in await self.children_of(root):
            found = await self.find(child, session_id)
            if found is not None:
                return found
        return None

    async def stop_tree(self, root: Session) -> None:
        for child in await self.children_of(root):
            await self.stop_tree(child)
        if isinstance(root, AgentSession):
            if root.runner is not None:
                await root.runner.stop()

    async def delete_tree(self, root: Session) -> None:
        await self.stop_tree(root)
        await self.session_store.delete_session(root.path)
        self._forget_tree(root.path)

    async def remove_child(self, parent: Session, child: Session | str) -> None:
        """Detach and delete a direct child subtree."""
        child_id = child.id if isinstance(child, Session) else child
        child_session = (
            child if isinstance(child, Session) else await self.resolve(child, parent)
        )
        if child_session is not None:
            await self.delete_tree(child_session)
        if child_id in parent.children:
            parent.children.remove(child_id)
            await self.save_session(parent)


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
