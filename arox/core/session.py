from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)

from arox.core.types import USER_INPUT_ID_KEY, ClientInput, MessagePayload

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime
    from arox.core.config import ConfigLoader
    from arox.core.io import AbstractIOAdapter

logger = logging.getLogger(__name__)


MODEL_MESSAGE_ID_KEY = "arox_model_message_id"


class SessionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    agent_name: str = ""


class StepEvent(SessionEvent):
    event_type: Literal["agent_step"] = "agent_step"
    input_event_id: str | None = None
    model_message_ids: list[str] = Field(default_factory=list)


class CommandCompletedEvent(SessionEvent):
    event_type: Literal["command_completed"] = "command_completed"
    client_input: ClientInput
    status: Literal["handled", "not_command", "unknown", "invalid", "error"]
    output: str | None = None
    error: str | None = None


class UserInputEvent(SessionEvent):
    event_type: Literal["user_input"] = "user_input"
    client_input: ClientInput


class ErrorEvent(SessionEvent):
    event_type: Literal["error"] = "error"
    error: str = ""


class CompactionEvent(SessionEvent):
    event_type: Literal["compaction"] = "compaction"
    step_boundary: bool = False
    trigger: Literal["manual", "token_threshold", "tool_request"] = "manual"
    llm_context_id: str = ""


AnySessionEvent = Annotated[
    Union[
        StepEvent,
        CommandCompletedEvent,
        UserInputEvent,
        ErrorEvent,
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

    def mark_dirty(self) -> None:
        if self.manager:
            self.manager.notify_dirty(self)

    def add_event(
        self,
        event: AnySessionEvent,
    ) -> AnySessionEvent:
        self.events.append(event)
        self.mark_dirty()
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
    agent_source: str = "runtime"
    workspace: str | None = None
    task_name: str | None = None
    target: str | None = None
    initial_message: str | None = None
    run_info: AgentRunInfo = Field(default_factory=AgentRunInfo)
    archived_message_histories: list[MessageHistorySegment] = Field(
        default_factory=list
    )
    message_history: MessageHistorySegment = Field(
        default_factory=MessageHistorySegment
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    runtime: Any = Field(default=None, exclude=True, repr=False)
    _runtime_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _ensure_runtime_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def task_id(self) -> str:
        return self.id

    @property
    def is_active(self) -> bool:
        return self.runtime is not None

    @property
    def is_empty(self) -> bool:
        return not (
            self.events
            or self.children
            or self.archived_message_histories
            or self.message_history.messages
            or self.task_name
            or self.target
            or self.initial_message
        )

    async def ensure_runtime(
        self,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
        runtime_factory: Callable[
            [ConfigLoader, AbstractIOAdapter, AgentSession], AgentRuntime
        ]
        | None = None,
    ) -> AgentRuntime:
        """Return the active runtime, starting it exactly once if necessary."""
        async with self._ensure_runtime_lock:
            if self.runtime is not None:
                return self.runtime

            if runtime_factory is None:
                from arox.core.agent_runtime import AgentRuntime

                runtime_factory = AgentRuntime

            runtime = runtime_factory(config_loader, io_adapter, self)
            await runtime.__aenter__()
            return runtime

    async def create_child_session(
        self,
        agent_name: str,
        *,
        agent_source: str = "child",
        workspace: Path | str | None = None,
        task_name: str | None = None,
        target: str | None = None,
        initial_message: str | None = None,
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
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
            owner=self,
            manager=self.manager,
        )
        self.children.append(child.id)
        if self.manager:
            self.manager._track(self, self.owner)
            self.manager._track(child, self)
            await self.manager.persist(self, child)
        if self.runtime:
            await self.runtime.broadcast_session_tree()
        return child

    def replace_message_history(self, messages: Sequence[ModelMessage]) -> None:
        self.message_history.messages = list(messages)

    @staticmethod
    def _ensure_model_message_id(message: ModelMessage) -> str:
        metadata = dict(message.metadata or {})
        message_id = metadata.get(MODEL_MESSAGE_ID_KEY)
        if not isinstance(message_id, str):
            message_id = uuid.uuid4().hex
            metadata[MODEL_MESSAGE_ID_KEY] = message_id
            message.metadata = metadata
        return message_id

    def _stored_model_messages(self) -> dict[str, ModelMessage]:
        messages: dict[str, ModelMessage] = {}
        for history in [*self.archived_message_histories, self.message_history]:
            for message in history.messages:
                message_id = (message.metadata or {}).get(MODEL_MESSAGE_ID_KEY)
                if isinstance(message_id, str):
                    messages[message_id] = message
        return messages

    def build_io_timeline(
        self,
    ) -> tuple[ModelMessage | CommandCompletedEvent | CompactionEvent, ...]:
        stored_messages = self._stored_model_messages()
        timeline: list[ModelMessage | CommandCompletedEvent | CompactionEvent] = []
        rendered_user_input_ids: set[str] = set()

        for event in self.events:
            if isinstance(event, UserInputEvent):
                rendered_user_input_ids.add(event.id)
                payload = event.client_input.payload
                if isinstance(payload, MessagePayload) and payload.content is not None:
                    timeline.append(
                        ModelRequest(
                            parts=[
                                UserPromptPart(
                                    content=payload.content,
                                    timestamp=event.timestamp,
                                )
                            ],
                            metadata={MODEL_MESSAGE_ID_KEY: event.id},
                        )
                    )
            elif isinstance(event, StepEvent):
                for message_id in event.model_message_ids:
                    message = stored_messages.get(message_id)
                    if message is None:
                        continue
                    if rendered_user_input_ids.intersection(
                        MessageHistorySegment._user_input_ids_on(message)
                    ):
                        continue
                    timeline.append(message)
            elif isinstance(event, CommandCompletedEvent):
                timeline.append(event)
            elif isinstance(event, CompactionEvent):
                timeline.append(event)
            elif isinstance(event, ErrorEvent):
                timeline.append(
                    ModelResponse(
                        parts=[TextPart(content=event.error)],
                        timestamp=event.timestamp,
                        metadata={MODEL_MESSAGE_ID_KEY: event.id},
                    )
                )

        return tuple(timeline)

    def build_io_snapshot(
        self, *, include_commands: bool = True
    ) -> tuple[ModelMessage, ...]:
        snapshot: list[ModelMessage] = []
        for item in self.build_io_timeline():
            if isinstance(item, CompactionEvent):
                continue
            if isinstance(item, CommandCompletedEvent):
                if not include_commands:
                    continue
                payload = item.client_input.payload
                assert not isinstance(payload, MessagePayload)
                command = payload.command
                display_text = (
                    command
                    if isinstance(command, str)
                    else json.dumps(command, ensure_ascii=False, sort_keys=True)
                )
                snapshot.append(
                    ModelRequest(
                        parts=[
                            UserPromptPart(
                                content=display_text,
                                timestamp=item.timestamp,
                            )
                        ]
                    )
                )
                content = item.output or item.error
                if content:
                    snapshot.append(
                        ModelResponse(
                            parts=[TextPart(content=content)],
                            timestamp=item.timestamp,
                        )
                    )
            else:
                snapshot.append(item)

        return tuple(snapshot)

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
        *,
        input_event_id: str | None,
        new_messages: Sequence[ModelMessage],
    ) -> StepEvent:
        message_ids = [
            self._ensure_model_message_id(message) for message in new_messages
        ]

        self.replace_message_history(message_history)
        event = StepEvent(
            agent_name=self.agent_name,
            input_event_id=input_event_id,
            model_message_ids=message_ids,
        )
        self.add_event(event)
        return event

    def record_command_completed(
        self,
        client_input: ClientInput,
        status: Literal["handled", "not_command", "unknown", "invalid", "error"],
        *,
        output: str | None = None,
        error: str | None = None,
    ) -> CommandCompletedEvent:
        input_id = client_input.server_message_id
        assert input_id is not None
        event = CommandCompletedEvent(
            id=input_id,
            client_input=client_input,
            status=status,
            output=output,
            error=error,
            agent_name=self.agent_name,
        )
        self.add_event(event)
        return event

    def record_user_input(self, client_input: ClientInput) -> None:
        input_id = client_input.server_message_id
        assert input_id is not None
        payload = client_input.payload
        assert isinstance(payload, MessagePayload)

        self.add_event(
            UserInputEvent(
                id=input_id, client_input=client_input, agent_name=self.agent_name
            )
        )
        last_user_messages = self.metadata.get("last_user_messages", [])
        text = payload.text_content
        if text:
            last_user_messages.append(text)
        self.metadata["last_user_messages"] = last_user_messages[-2:]

    @staticmethod
    def format_error(error: BaseException | str) -> str:
        if isinstance(error, asyncio.CancelledError):
            return "Task interrupted."
        if isinstance(error, str):
            return error
        return f"{type(error).__name__}: {error!s}"

    def record_error_event(self, error: BaseException | str) -> str:
        """Record an error event without changing task scheduling state."""
        err_msg = self.format_error(error)
        self.add_event(ErrorEvent(error=err_msg, agent_name=self.agent_name))
        return err_msg

    def record_compaction(
        self,
        compacted_messages: Sequence[ModelMessage],
        step_boundary: bool,
        llm_context_id: str,
        *,
        trigger: Literal["manual", "token_threshold", "tool_request"] = "manual",
        previous_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        history_to_archive = (
            self.message_history.messages
            if previous_messages is None
            else previous_messages
        )
        for message in [*history_to_archive, *compacted_messages]:
            self._ensure_model_message_id(message)

        self.run_info.llm_context_id = llm_context_id
        self._start_message_history(
            compacted_messages, previous_messages=previous_messages
        )
        self.add_event(
            CompactionEvent(
                step_boundary=step_boundary,
                trigger=trigger,
                llm_context_id=llm_context_id,
                agent_name=self.agent_name,
            )
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
        self,
        event_id: str | None,
        new_owner: Session | None = None,
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
            update={"manager": None, "owner": None, "runtime": None}
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
                "manager": manager_ref,
                "owner": owner,
                "runtime": None,
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
                await sub_session.fork_at(None, new_session)
            except Exception:
                logger.warning(
                    "Failed to fork child session %s", sub_session.id, exc_info=True
                )

        if self.manager:
            self.manager._track(new_session, owner)
            new_session.mark_dirty()
            if owner:
                owner.mark_dirty()

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
                await self.persist(session)
            except Exception as e:
                logger.error(f"Failed to save session {session.id}: {e}", exc_info=True)
                # Re-queue for retry on the next cycle.
                self._dirty_sessions.setdefault(session.id, session)

    async def persist(self, *sessions: Session) -> None:
        """Persist sessions immediately, bypassing the dirty debounce."""
        for session in sessions:
            self._dirty_sessions.pop(session.id, None)
            self._track(session, session.owner)
            if (
                isinstance(session, AgentSession)
                and len(session.path) == 1
                and session.is_empty
            ):
                continue
            await self.session_store.save_session(session)

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
        """Stop every active tree, including children whose root is inactive."""
        active_sessions = [
            session
            for session in self._sessions.values()
            if isinstance(session, AgentSession) and session.is_active
        ]
        active_paths = {tuple(session.path) for session in active_sessions}
        roots = [
            session
            for session in active_sessions
            if not any(
                tuple(session.path[:depth]) in active_paths
                for depth in range(1, len(session.path))
            )
        ]
        await asyncio.gather(
            *(self.stop_tree(session) for session in roots),
            return_exceptions=True,
        )

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
        await self.stop_descendants(root)
        if isinstance(root, AgentSession):
            if root.runtime is not None:
                await root.runtime.close()

    async def stop_descendants(self, root: Session) -> None:
        """Stop all active descendants in child-first order, preserving root."""
        for child in await self.children_of(root):
            await self.stop_tree(child)

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
            await self.persist(parent)


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
