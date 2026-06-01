import logging
from dataclasses import dataclass
from typing import ClassVar

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import (
    AgentSession,
    Session,
    UserInputEvent,
)

logger = logging.getLogger(__name__)


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


def user_turns_from_session(session: AgentSession) -> list[tuple[str, str]]:
    """List ``(input_id, text)`` for every user turn present in session."""
    turns: list[tuple[str, str]] = []
    for event in session.events:
        if isinstance(event, UserInputEvent):
            turns.append((event.id, event.text))
    return turns


class SessionPlugin(Plugin):
    """Session command layer.

    Core session binding, persistence and event recording live on
    ``LLMBaseAgent``. This plugin keeps the `/fork` command and exposes small
    compatibility proxies for older tests and integrations.
    """

    def __init__(self, agent):
        super().__init__(agent)

    def commands(self):
        return [CommandSpec(ForkEvent, self.handle_fork, self.complete_fork)]

    async def complete_fork(self, req: CompletionRequest):
        session = getattr(self.agent, "session", None)
        if not session:
            return
        turns = user_turns_from_session(session)
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

    async def handle_fork(self, event: ForkEvent) -> str:
        agent_session = self.agent.session

        if not event.event_id:
            return "Cannot fork: specify a user turn."

        try:
            new_agent_session = agent_session.fork_at(
                event.event_id, agent_session.owner
            )
        except ValueError as e:
            return str(e)

        await self._fork_children(agent_session, new_agent_session)

        # Save the new branch (now with its children linked).
        await new_agent_session.save()

        return (
            f"Forked at event {event.event_id}. New branch session id: {new_agent_session.id}\n"
            f"Resume with: --resume {new_agent_session.id}"
        )

    async def _fork_children(
        self,
        agent_session: AgentSession,
        new_agent_session: Session | None = None,
    ) -> None:
        """Persist an empty fork of each subsession beneath ``owner_path``."""
        for child_id in agent_session.children:
            try:
                if agent_session.manager:
                    sub_session = await agent_session.manager.load_session(
                        child_id, agent_session
                    )
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

            forked = sub_session.fork_at(None, new_agent_session)
            await self._fork_children(sub_session, forked)
            await forked.save()
