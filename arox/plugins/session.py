import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field
from pydantic_ai.messages import ModelMessage

from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import (
    FileSessionStore,
    Session,
    _deserialize_messages,
    _serialize_messages,
)
from arox.plugins.slots import (
    AGENT_COMMAND,
    AGENT_ERROR,
    AGENT_RESET,
    AGENT_STEP,
    AGENT_STEP_FAILURE,
    RECORD_EVENT,
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
                raw = event.data.get("new_messages", [])
                history.extend(_deserialize_messages(raw))
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

    def user_turn_anchors(self) -> list[int]:
        return [i for i, ev in enumerate(self.events) if ev.event_type == "user_input"]

    def resolve_user_turn(self, n: int) -> int | None:
        if n < 1:
            return None
        anchors = self.user_turn_anchors()
        if len(anchors) < n:
            return None
        return anchors[-n]

    def truncated_copy(self, event_index: int) -> "AgentSession":
        return AgentSession(
            id=str(uuid.uuid4()),
            agent_name=self.agent_name,
            owner_id=self.owner_id,
            owner_path=list(self.owner_path),
            events=[ev.model_copy(deep=True) for ev in self.events[:event_index]],
            extra=dict(self.extra),
            forked_from={self.agent_name: event_index},
        )


@dataclass(kw_only=True)
class ForkEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("fork",)
    description: ClassVar[str] = (
        "Fork the session at a user turn - /fork [N] (relative) or /fork @<index> (absolute)"
    )

    n: int | None = 1
    event_index: int | None = None

    @classmethod
    def from_slash(cls, name, arg):
        raw = (arg or "").strip()
        if not raw:
            return cls(n=1)
        if raw.startswith("@"):
            try:
                return cls(n=None, event_index=int(raw[1:]))
            except ValueError:
                return cls(n=1)
        try:
            return cls(n=max(int(raw), 1))
        except ValueError:
            return cls(n=1)


class SessionPlugin(Plugin):
    """Manages session persistence and forking for the agent and its subagents."""

    def __init__(self, agent):
        super().__init__(agent)
        self.session_store = FileSessionStore(
            max_age_days=self.agent.parsed_config.app.session_max_age_days
        )
        self.app_session = None
        self.agent_session = None

    async def _get_subagents(self):
        return [
            a for a in await self.agent.invoke_slot(SUBAGENTS) or [] if a != self.agent
        ]

    def commands(self):
        return [CommandSpec(ForkEvent, self.handle_fork)]

    def subscribe(self):
        return [
            (AGENT_RESET, self.on_agent_reset),
            (AGENT_STEP, self.on_agent_step),
            (AGENT_STEP_FAILURE, self.on_agent_step_failure),
            (AGENT_COMMAND, self.on_agent_command),
            (USER_INPUT, self.on_user_input),
            (AGENT_ERROR, self.on_error),
            (RECORD_EVENT, self.on_event),
        ]

    async def on_start(self):
        await self.session_store.cleanup()

        # The host app should have set self.agent.app_session
        self.app_session = getattr(self.agent, "app_session", None)

        if self.app_session:
            # Initialize MainAgent's AgentSession
            agent_session_id = self.app_session.metadata.get(
                f"{self.agent.name}_session_id"
            )
            if agent_session_id:
                loaded_agent_session = await self.session_store.load_session(
                    agent_session_id, owner_path=[self.app_session.id]
                )
                if loaded_agent_session and isinstance(
                    loaded_agent_session, AgentSession
                ):
                    self.restore_agent_session(loaded_agent_session)
                else:
                    await self._create_fresh_agent_session(
                        self.app_session.id, [self.app_session.id]
                    )
            else:
                await self._create_fresh_agent_session(
                    self.app_session.id, [self.app_session.id]
                )
                assert self.agent_session is not None
                self.app_session.metadata[f"{self.agent.name}_session_id"] = (
                    self.agent_session.id
                )
                await self.session_store.save_session(self.app_session)
        else:
            # Fallback if no app_session is provided (e.g. subagents or tests)
            # Subagents should have their owner_id set by the parent agent during creation,
            # but for now we just create a fresh one if it doesn't exist.
            if not self.agent_session:
                await self._create_fresh_agent_session("unknown", ["unknown"])

    def restore_agent_session(self, agent_session: AgentSession):
        self.agent_session = agent_session
        self.agent.message_history = agent_session.rebuild_message_history()
        restored_id = agent_session.rebuild_llm_context_id()
        if restored_id:
            self.agent.llm_context_id = restored_id
        if self.agent.model_ref:
            self.agent.set_model(self.agent.model_ref)

    async def _create_fresh_agent_session(self, owner_id: str, owner_path: list[str]):
        self.agent_session = AgentSession(
            agent_name=self.agent.name, owner_id=owner_id, owner_path=owner_path
        )
        await self.agent.reset()

    async def on_stop(self):
        await self.save()

    async def on_agent_reset(self) -> None:
        if self.agent_session:
            self.agent_session.add_event(
                "reset", {"llm_context_id": self.agent.llm_context_id}
            )

    async def on_agent_step(self, input_content: str | None, result: Any) -> None:
        if self.agent_session:
            new_messages = result.new_messages()
            usage = result.usage
            self.agent_session.add_event(
                "agent_step",
                {
                    "input": input_content,
                    "new_messages": _serialize_messages(new_messages),
                    "request_tokens": usage.input_tokens if usage else None,
                    "response_tokens": usage.output_tokens if usage else None,
                },
            )

    async def on_agent_step_failure(
        self, input_content: str | None, messages: list[ModelMessage]
    ) -> None:
        if self.agent_session:
            prev_len = len(self.agent.message_history)
            new_messages = messages[prev_len:]
            if new_messages:
                self.agent_session.add_event(
                    "agent_step",
                    {
                        "input": input_content,
                        "new_messages": _serialize_messages(new_messages),
                        "request_tokens": None,
                        "response_tokens": None,
                    },
                )

    async def on_agent_command(self, command: str, arg: str | None) -> None:
        if self.agent_session:
            self.agent_session.add_event("command", {"command": command, "arg": arg})

    async def on_user_input(self, text: str, client_message_id: str | None) -> None:
        if self.agent_session:
            self.agent_session.add_event(
                "user_input",
                {
                    "text": text,
                    "client_message_id": client_message_id,
                },
            )

    async def on_error(self, error: Exception) -> None:
        if self.agent_session:
            self.agent_session.add_event(
                "error", {"error": f"{type(error).__name__}: {error!s}"}
            )

    async def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.agent_session:
            self.agent_session.add_event(event_type, data)

    async def save(self):
        # Save current agent's session
        if self.agent_session:
            # Update last_user_messages for MainAgent
            if self.app_session:
                last_user_messages = []
                if hasattr(self.agent, "message_history"):
                    from pydantic_ai.messages import ModelRequest, UserPromptPart

                    for msg in reversed(self.agent.message_history):
                        if isinstance(msg, ModelRequest):
                            for part in msg.parts:
                                if isinstance(part, UserPromptPart):
                                    content = part.content
                                    if isinstance(content, str):
                                        last_user_messages.append(content)
                                    elif isinstance(content, (list, tuple)):
                                        text_parts = [
                                            c for c in content if isinstance(c, str)
                                        ]
                                        if text_parts:
                                            last_user_messages.append(
                                                " ".join(text_parts)
                                            )
                                    if len(last_user_messages) >= 2:
                                        break
                        if len(last_user_messages) >= 2:
                            break
                if last_user_messages:
                    self.app_session.metadata["last_user_messages"] = list(
                        reversed(last_user_messages)
                    )
                await self.session_store.save_session(self.app_session)

            await self.session_store.save_session(self.agent_session)

        # Broadcast to subagents
        for subagent in await self._get_subagents():
            if subagent_plugin := subagent.get_plugin(SessionPlugin):
                await subagent_plugin.save()

    async def handle_fork(self, event: ForkEvent) -> str:
        if not self.app_session:
            return "Cannot fork: not a main agent or no app session."

        agent_session = self.agent_session
        if not isinstance(agent_session, AgentSession):
            return "Cannot fork: invalid agent session."

        if event.event_index is not None:
            target = event.event_index
            anchors = set(agent_session.user_turn_anchors())
            if target not in anchors:
                return f"Cannot fork at @{target}: not a user-turn anchor."
        else:
            n = event.n or 1
            resolved = agent_session.resolve_user_turn(n)
            if resolved is None:
                return f"Cannot fork {n} user turn(s) back: not enough history."
            target = resolved

        new_app_session = self.app_session.fork_at(self.agent.name, target)
        new_agent_session = agent_session.truncated_copy(target)
        new_agent_session.owner_id = new_app_session.id
        new_agent_session.owner_path = [new_app_session.id]

        new_app_session.metadata[f"{self.agent.name}_session_id"] = new_agent_session.id

        self.app_session.add_event(
            "fork",
            {
                "agent_name": self.agent.name,
                "event_index": target,
                "new_session_id": new_app_session.id,
            },
        )
        await self.session_store.save_session(self.app_session)

        # Save new sessions
        await self.session_store.save_session(new_app_session)
        await self.session_store.save_session(new_agent_session)

        # Broadcast fork/reset to subagents
        for subagent in await self._get_subagents():
            if subagent_plugin := subagent.get_plugin(SessionPlugin):
                await subagent_plugin.reset_for_fork(
                    new_app_session.id, [new_app_session.id, new_agent_session.id]
                )

        return (
            f"Forked at event @{target}. New branch session id: {new_app_session.id}\n"
            f"Resume with: --resume {new_app_session.id}"
        )

    async def reset_for_fork(self, owner_id: str, owner_path: list[str]):
        await self._create_fresh_agent_session(owner_id, owner_path)
        assert self.agent_session is not None
        await self.session_store.save_session(self.agent_session)

        for subagent in await self._get_subagents():
            if subagent_plugin := subagent.get_plugin(SessionPlugin):
                await subagent_plugin.reset_for_fork(
                    self.agent_session.id,
                    owner_path + [self.agent_session.id],
                )
