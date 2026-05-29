import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextContent,
    UserPromptPart,
)

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import (
    USER_INPUT_ID_KEY,
    Session,
    SessionStore,
    _deserialize_messages,
    _serialize_messages,
    derive_child_session_id,
    register_session_type,
)
from arox.plugins.slots import (
    AGENT_COMMAND,
    AGENT_ERROR,
    AGENT_RESET,
    AGENT_SESSION,
    AGENT_STEP,
    AGENT_STEP_FAILURE,
    RECORD_EVENT,
    SESSION_STORE,
    SET_SESSION,
    SUBAGENTS,
    USER_INPUT,
)

logger = logging.getLogger(__name__)


class AgentSession(Session):
    session_type: str = "agent"
    agent_name: str
    extra: dict[str, Any] = Field(default_factory=dict)

    def rebuild_message_history(self) -> list[ModelMessage]:
        history: list[ModelMessage] = []
        for event in self.events:
            if event.event_type == "agent_step":
                history.extend(
                    _deserialize_messages(event.data.get("new_messages", []))
                )
            elif event.event_type == "compaction":
                raw = event.data.get("compacted_messages", [])
                history = _deserialize_messages(raw)
        return history

    def rebuild_llm_context_id(self) -> str | None:
        context_id = None
        for event in self.events:
            if event.event_type in ("compaction", "reset"):
                context_id = event.data.get("llm_context_id")
        return context_id

    def fork_at(self, event_id: str | None, owner_path: list[str]) -> "AgentSession":
        """Branch a new session off this one, nested under ``owner_path``.

        With ``event_id`` the new session is a truncated copy holding the events
        up to (but excluding) that anchor, tagged via ``forked_from``. With
        ``event_id`` set to ``None`` it starts empty and unforked.

        ``owner_path`` is the chain of owner session ids the branch nests under
        (empty roots it at the top level); the owner id is corrected to its last
        element rather than inherited from this session. A nested branch derives
        its id from the owner so the owner re-derives the same id on resume; a
        root branch gets a fresh uuid.

        Raises ``ValueError`` if ``event_id`` is given but not found.
        """
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
        owner_path = list(owner_path)
        owner_id = owner_path[-1] if owner_path else None
        new_id = (
            derive_child_session_id(owner_id, self.agent_name)
            if owner_id
            else str(uuid.uuid4())
        )
        return AgentSession(
            id=new_id,
            agent_name=self.agent_name,
            owner_id=owner_id,
            owner_path=owner_path,
            events=events,
            extra=dict(self.extra),
            forked_from=forked_from,
        )

    @staticmethod
    def user_input_id_of(message: ModelMessage) -> str | None:
        """Return the user-turn id tagged on ``message`` via ``USER_INPUT_ID_KEY``.

        The id lives on the user prompt's ``TextContent`` metadata and is the
        canonical fork anchor; returns ``None`` for non-user messages.
        """
        if not isinstance(message, ModelRequest):
            return None
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            content = part.content
            items = content if isinstance(content, (list, tuple)) else [content]
            for item in items:
                if isinstance(item, TextContent) and isinstance(item.metadata, dict):
                    input_id = item.metadata.get(USER_INPUT_ID_KEY)
                    if input_id:
                        return input_id
        return None

    @staticmethod
    def _user_message_text(message: ModelMessage) -> str:
        if not isinstance(message, ModelRequest):
            return ""
        chunks: list[str] = []
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            content = part.content
            items = content if isinstance(content, (list, tuple)) else [content]
            for item in items:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, TextContent):
                    chunks.append(item.content)
        return "".join(chunks).strip()

    @classmethod
    def user_turns_from_history(
        cls, history: Sequence[ModelMessage]
    ) -> list[tuple[str, str]]:
        """List ``(input_id, text)`` for every user turn present in ``history``.

        Turns whose context was dropped by compaction are absent, so the result
        stays aligned 1:1 with the user messages actually in ``history``.
        """
        turns: list[tuple[str, str]] = []
        for message in history:
            input_id = cls.user_input_id_of(message)
            if input_id:
                turns.append((input_id, cls._user_message_text(message)))
        return turns


register_session_type(AgentSession)


@dataclass(kw_only=True)
class ForkEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("fork",)
    description: ClassVar[str] = (
        "Fork the session at a user turn - /fork <event_id> (press Tab to choose)"
    )

    event_id: str | None = None

    @classmethod
    def from_slash(cls, name, arg):
        raw = (arg or "").strip()
        if raw.startswith("@"):
            raw = raw[1:].strip()
        return cls(event_id=raw or None)


class SessionPlugin(Plugin):
    """Manages session persistence and forking for the agent and its subagents."""

    def __init__(self, agent):
        super().__init__(agent)
        self.agent_session = None
        # All configured via the SET_SESSION slot before ``on_start`` by the
        # owning App / SubagentPlugin: the shared session store, the session id
        # to resume, and the chain of owner session ids it nests under (empty
        # for the root).
        self.session_store: SessionStore | None = None
        self._session_id: str | None = None
        self._session_owner_path: list[str] = []

    @property
    def _store(self) -> SessionStore:
        assert self.session_store is not None, "SET_SESSION must run before on_start"
        return self.session_store

    async def _subagents_of(self, agent):
        return [a for a in await agent.invoke_slot(SUBAGENTS) or [] if a != agent]

    async def _get_subagents(self):
        return await self._subagents_of(self.agent)

    def commands(self):
        return [CommandSpec(ForkEvent, self.handle_fork, self.complete_fork)]

    async def complete_fork(self, req: CompletionRequest):
        history = getattr(self.agent, "message_history", None) or []
        turns = AgentSession.user_turns_from_history(history)
        typed = req.current_token.lstrip("@").lower()
        total = len(turns)
        for back, (input_id, text) in enumerate(reversed(turns), start=1):
            if typed and typed not in input_id.lower():
                continue
            text = text.replace("\n", " ")
            if len(text) > 40:
                text = text[:40] + "…"
            turn_no = total - back + 1
            label = f"@{back} (turn {turn_no})"
            if text:
                label = f"{label}: {text}"
            yield CompletionItem(
                value=input_id,
                label=label,
                group="fork",
                score=float(total - back),
            )

    def on_load(self):
        self.agent.provide_slot(AGENT_SESSION, lambda: self.agent_session)
        self.agent.provide_slot(SESSION_STORE, lambda: self.session_store)
        self.agent.provide_slot(SET_SESSION, self.on_set_session)
        self.agent.provide_slot(AGENT_RESET, self.on_agent_reset)
        self.agent.provide_slot(AGENT_STEP, self.on_agent_step)
        self.agent.provide_slot(AGENT_STEP_FAILURE, self.on_agent_step_failure)
        self.agent.provide_slot(AGENT_COMMAND, self.on_agent_command)
        self.agent.provide_slot(USER_INPUT, self.on_user_input)
        self.agent.provide_slot(AGENT_ERROR, self.on_error)
        self.agent.provide_slot(RECORD_EVENT, self.on_event)

    def on_set_session(
        self,
        session_id: str | None,
        owner_path: list[str] | None,
        session_store: SessionStore,
    ):
        """Configure the session store, id to resume and owner path it nests under.

        Called via the :data:`SET_SESSION` slot before :meth:`on_start`. The
        main agent's session is the root (empty ``owner_path``); a subagent's
        session is nested under its owner (see :class:`SubagentPlugin`).
        """
        self.session_store = session_store
        self._session_id = session_id
        self._session_owner_path = list(owner_path or [])

    async def on_start(self):
        # Both the root and nested subagent sessions follow the same rule: resume
        # ``session_id`` under ``owner_path`` when it exists on disk, otherwise
        # create it there. The root simply has an empty owner path and no owner.
        if self.agent_session:
            return
        owner_path = list(self._session_owner_path)
        owner_id = owner_path[-1] if owner_path else None
        # Only the root prunes expired sessions; nested subagents share the same
        # store and would just re-scan the same top-level dirs redundantly.
        if owner_id is None:
            await self._store.cleanup()
        restore_id = self._session_id
        if restore_id:
            loaded = await self._store.load_session(restore_id, owner_path)
            if loaded and isinstance(loaded, AgentSession):
                self.restore_agent_session(loaded)
                return
        await self._create_fresh_agent_session(
            owner_id, owner_path, session_id=restore_id
        )

    def restore_agent_session(self, agent_session: AgentSession):
        self.agent_session = agent_session
        self.agent.message_history = agent_session.rebuild_message_history()
        restored_id = agent_session.rebuild_llm_context_id()
        if restored_id:
            self.agent.llm_context_id = restored_id
        if self.agent.model_ref:
            self.agent.set_model(self.agent.model_ref)

    async def _create_fresh_agent_session(
        self,
        owner_id: str | None,
        owner_path: list[str],
        session_id: str | None = None,
    ):
        kwargs: dict[str, Any] = {
            "agent_name": self.agent.name,
            "owner_id": owner_id,
            "owner_path": owner_path,
        }
        if session_id:
            kwargs["id"] = session_id
        self.agent_session = AgentSession(**kwargs)
        await self.agent.reset()

    async def on_stop(self):
        await self.save()

    async def on_agent_reset(self) -> None:
        if self.agent_session:
            self.agent_session.add_event(
                "reset", {"llm_context_id": self.agent.llm_context_id}
            )

    async def on_agent_step(self, result: Any) -> None:
        if self.agent_session:
            new_messages = result.new_messages()
            usage = result.usage
            self.agent_session.add_event(
                "agent_step",
                {
                    "new_messages": _serialize_messages(new_messages),
                    "request_tokens": usage.input_tokens if usage else None,
                    "response_tokens": usage.output_tokens if usage else None,
                },
            )

    async def on_agent_step_failure(self, messages: list[ModelMessage]) -> None:
        if self.agent_session:
            prev_len = len(self.agent.message_history)
            new_messages = messages[prev_len:]
            if new_messages:
                self.agent_session.add_event(
                    "agent_step",
                    {
                        "new_messages": _serialize_messages(new_messages),
                        "request_tokens": None,
                        "response_tokens": None,
                    },
                )

    async def on_agent_command(self, command: str, arg: str | None) -> None:
        if self.agent_session:
            self.agent_session.add_event("command", {"command": command, "arg": arg})

    async def on_user_input(self, text: str, input_id: str) -> None:
        if self.agent_session:
            self.agent_session.add_event("user_input", {"text": text}, id=input_id)

    async def on_error(self, error: Exception) -> None:
        if self.agent_session:
            self.agent_session.add_event(
                "error", {"error": f"{type(error).__name__}: {error!s}"}
            )

    async def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.agent_session:
            self.agent_session.add_event(event_type, data)

    def _last_user_messages(self, limit: int = 2) -> list[str]:
        """The most recent user-message texts, oldest first (for display)."""
        history = getattr(self.agent, "message_history", None) or []
        found: list[str] = []
        for msg in reversed(history):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if not isinstance(part, UserPromptPart):
                        continue
                    content = part.content
                    if isinstance(content, str):
                        found.append(content)
                    elif isinstance(content, (list, tuple)):
                        text_parts = [c for c in content if isinstance(c, str)]
                        if text_parts:
                            found.append(" ".join(text_parts))
                    if len(found) >= limit:
                        break
            if len(found) >= limit:
                break
        return list(reversed(found))

    async def save(self):
        if self.agent_session:
            last_user_messages = self._last_user_messages()
            if last_user_messages:
                self.agent_session.metadata["last_user_messages"] = last_user_messages

            await self._store.save_session(self.agent_session)

        # Broadcast to subagents
        for subagent in await self._get_subagents():
            if subagent_plugin := subagent.get_plugin(SessionPlugin):
                await subagent_plugin.save()

    async def handle_fork(self, event: ForkEvent) -> str:
        agent_session = self.agent_session
        if not isinstance(agent_session, AgentSession):
            return "Cannot fork: invalid agent session."

        if not event.event_id:
            return "Cannot fork: specify a user turn (press Tab to choose one)."

        target = agent_session.index_of_event(event.event_id)
        if target is None:
            return f"Cannot fork at {event.event_id}: event not found."
        if agent_session.events[target].event_type != "user_input":
            return f"Cannot fork at {event.event_id}: not a user-turn anchor."

        # Branch the session itself: a truncated copy is the new top-level
        # session, rooted (no owner) regardless of where the original sat.
        new_agent_session = agent_session.fork_at(event.event_id, [])
        await self._store.save_session(new_agent_session)

        # Re-root each subagent (and its nested subagents) under the new branch.
        await self._fork_subagents(self.agent, [new_agent_session.id])

        return (
            f"Forked at event @{target}. New branch session id: {new_agent_session.id}\n"
            f"Resume with: --resume {new_agent_session.id}"
        )

    async def _fork_subagents(self, agent, owner_path: list[str]) -> None:
        """Persist an empty fork of each subagent session beneath ``owner_path``.

        Each subagent's session is re-rooted under the new branch via
        :meth:`AgentSession.fork_at` and saved so a later resume re-derives it
        (``fork_at`` derives the child id from its owner); the live subagents
        keep their own sessions. Recurses so nested subagents nest beneath
        their forked owner. Subagents that never started have no session and
        are skipped — resume recreates them empty regardless.
        """
        for subagent in await self._subagents_of(agent):
            sub_session = await subagent.invoke_slot(AGENT_SESSION)
            if not sub_session:
                continue
            forked = sub_session.fork_at(None, owner_path)
            await self._store.save_session(forked)
            await self._fork_subagents(subagent, owner_path + [forked.id])
