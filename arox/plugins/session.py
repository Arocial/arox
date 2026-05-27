import logging
from dataclasses import dataclass
from typing import ClassVar

from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import AppSession, FileSessionStore
from arox.plugins.slots import ALL_AGENTS

logger = logging.getLogger(__name__)


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
        self.session = None

    def commands(self):
        return [CommandSpec(ForkEvent, self.handle_fork)]

    def _get_all_agents(self):
        agents = [self.agent]
        for provider in self.agent.get_slot(ALL_AGENTS):
            agents.extend(provider())
        return agents

    async def on_start(self):
        await self.session_store.cleanup()

        # The session_id should be passed to the agent somehow, e.g., via an attribute
        # set by the App before calling run(). Let's assume self.agent.session_id exists.
        session_id = getattr(self.agent, "session_id", None)

        restored = False
        if session_id:
            loaded = await self.session_store.load_session(session_id)
            if loaded:
                self.session = loaded
                restored = True
                await self.agent.agent_io.send(f"Session restored: {self.session.id}")

        if not restored:
            self.session = AppSession.create(
                self.agent.name, workspace=str(self.agent.workspace)
            )

        assert self.session is not None
        for agent in self._get_all_agents():
            agent.restore_session(self.session.get_agent_session(agent.name))

    async def on_stop(self):
        await self._save_session()

    async def _save_session(self):
        if not self.session:
            return

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
                                text_parts = [c for c in content if isinstance(c, str)]
                                if text_parts:
                                    last_user_messages.append(" ".join(text_parts))
                            if len(last_user_messages) >= 2:
                                break
                if len(last_user_messages) >= 2:
                    break

        if last_user_messages:
            self.session.metadata["last_user_messages"] = list(
                reversed(last_user_messages)
            )

        await self.session_store.save_session(self.session)

    async def handle_fork(self, event: ForkEvent) -> str:
        agent_session = self.agent.agent_session
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
        new_id = await self.fork_session(self.agent.name, target)
        return (
            f"Forked at event @{target}. New branch session id: {new_id}\n"
            f"Resume with: --resume {new_id}"
        )

    async def fork_session(self, agent_name: str, event_index: int) -> str:
        assert self.session is not None
        new_session = self.session.fork_at(agent_name, event_index)
        self.session.add_event(
            "fork",
            {
                "agent_name": agent_name,
                "event_index": event_index,
                "new_session_id": new_session.id,
            },
        )
        await self._save_session()
        await self.session_store.save_session(new_session)
        return new_session.id
