import logging
from dataclasses import dataclass
from typing import ClassVar

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import AgentSession, UserInputEvent
from arox.plugins.slots import AGENT_INFO

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SetModelEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("model",)
    description: ClassVar[str] = "Switch LLM model - /model <model_name>"

    model_ref: str

    @classmethod
    def from_slash(cls, name, arg):
        return cls(model_ref=arg or "")


@dataclass(kw_only=True)
class InfoEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("info",)
    description: ClassVar[str] = "Show current chat files and model in use - /info"


@dataclass(kw_only=True)
class ResetEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("reset",)
    description: ClassVar[str] = "Reset chat history and chat files - /reset"


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


class CorePlugin(Plugin):
    def commands(self):
        return [
            CommandSpec(SetModelEvent, self.handle_set_model, self.complete_model_ref),
            CommandSpec(InfoEvent, self.handle_info),
            CommandSpec(ResetEvent, self.handle_reset),
            CommandSpec(ForkEvent, self.handle_fork, self.complete_fork),
        ]

    async def handle_set_model(self, event: SetModelEvent) -> str:
        if not event.model_ref:
            return "Please specify a model name"
        self.agent.set_model(event.model_ref)
        return f"Model switched to {event.model_ref}"

    async def complete_model_ref(self, req: CompletionRequest):
        typed = req.current_token.lower()
        for ref in self.agent.parsed_config.available_models:
            if typed and typed not in ref.lower():
                continue
            score = 2.0 if ref.lower().startswith(typed) else 1.0 if typed else 0.0
            yield CompletionItem(
                value=ref,
                label=ref,
                group="model",
                score=score,
            )

    async def handle_info(self, event: InfoEvent) -> str:
        lines = [f"Current model: {getattr(self.agent, 'provider_model', 'Unknown')}"]
        for info in await self.agent.invoke_slot(AGENT_INFO) or []:
            if info:
                lines.append(info)
        return "\n".join(lines)

    async def handle_reset(self, event: ResetEvent) -> str:
        await self.agent.reset()
        return "Reset complete."

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
            new_agent_session = await agent_session.fork_at(
                event.event_id, agent_session.owner
            )
        except ValueError as e:
            return str(e)

        # Save the new branch (now with its children linked).
        await new_agent_session.save()

        return (
            f"Forked at event {event.event_id}. New branch session id: {new_agent_session.id}\n"
            f"Resume with: --resume {new_agent_session.id}"
        )
