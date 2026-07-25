import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic_ai import ModelMessage, ModelRequest, RunContext, UserPromptPart

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.core.session import AgentSession, UserInputEvent
from arox.plugins.slots import AGENT_INFO

if TYPE_CHECKING:
    from arox.core.llm_base import LLMBaseAgent

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


@dataclass(kw_only=True)
class SkillEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("skill",)
    description: ClassVar[str] = "Load skill to prompt - /skill <skill_name>"

    skills: list[str]

    @classmethod
    def from_slash(cls, name, arg):
        return cls(skills=arg.split() if arg else [])


def user_turns_from_session(session: AgentSession) -> list[tuple[str, str]]:
    """List ``(input_id, text)`` for every user turn present in session."""
    turns: list[tuple[str, str]] = []
    for event in session.events:
        if isinstance(event, UserInputEvent):
            text = event.user_input.text_content or ""
            turns.append((event.id, text))
    return turns


class CorePlugin(Plugin):
    def __init__(self, agent: Any):
        super().__init__(agent)
        self.agent: "LLMBaseAgent" = agent
        self._pending_skills: list[str] = []

    def commands(self):
        return [
            CommandSpec(SetModelEvent, self.handle_set_model, self.complete_model_ref),
            CommandSpec(InfoEvent, self.handle_info),
            CommandSpec(ResetEvent, self.handle_reset),
            CommandSpec(ForkEvent, self.handle_fork, self.complete_fork),
            CommandSpec(SkillEvent, self.handle_skill, self.complete_skill),
        ]

    async def handle_set_model(self, event: SetModelEvent) -> str:
        if not event.model_ref:
            return "Please specify a model name"
        self.agent.override_model(event.model_ref)
        return f"Model switched to {event.model_ref}"

    async def complete_model_ref(self, req: CompletionRequest):
        typed = req.current_token.lower()
        for ref in self.agent.config.available_models:
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
        self.agent.reload_config()
        lines = [
            f"Current model: {self.agent.provider_model or 'Unknown'}",
            f"Usage: total: {self.agent.run_info.total_tokens}, context: {self.agent.run_info.context_tokens}",
        ]
        for info in await self.agent.invoke_slot(AGENT_INFO) or []:
            if info:
                lines.append(info)
        return "\n".join(lines)

    async def handle_reset(self, event: ResetEvent) -> str:
        self._pending_skills = []
        await self.agent.reset()
        return "Reset complete."

    async def complete_fork(self, req: CompletionRequest):
        session = self.agent.session
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
            new_agent_session = await agent_session.fork_at(event.event_id)
        except ValueError as e:
            return str(e)

        # Save the new branch (now with its children linked).
        await new_agent_session.save()

        return (
            f"Forked at event {event.event_id}. New branch session id: {new_agent_session.id}\n"
            f"Resume with: --resume {new_agent_session.id}"
        )

    async def handle_skill(self, event: SkillEvent) -> str | None:
        if not event.skills:
            return "Please specify skills."
        loaded = []
        not_found = []
        for skill_name in event.skills:
            if skill_name in self.agent.config.skills:
                self._pending_skills.append(skill_name)
                loaded.append(skill_name)
            else:
                not_found.append(skill_name)

        msg = ""
        if loaded:
            msg += f"Loaded skills: {', '.join(loaded)}.\n"
        if not_found:
            msg += f"Skills not found: {', '.join(not_found)}.\n"
        return msg.strip() or None

    async def complete_skill(self, req: CompletionRequest):
        typed = req.current_token.lower()
        for skill_name in self.agent.config.skills:
            if typed and typed not in skill_name.lower():
                continue
            yield CompletionItem(
                value=skill_name,
                label=skill_name,
                group="skill",
                score=2.0 if skill_name.lower().startswith(typed) else 1.0,
            )

    async def history_processor(
        self, ctx: RunContext[Any], messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if not self._pending_skills:
            return messages

        if messages and isinstance(messages[-1], ModelRequest):
            extra_content = self.agent.build_skill_prompts(self._pending_skills)

            self._pending_skills = []

            if extra_content:
                text_part = (
                    "The following skills are manually loaded for reference:\n\n"
                    + "\n\n".join(extra_content)
                )
                new_request = ModelRequest(
                    parts=[UserPromptPart(content=text_part)],
                    metadata={"arox_internal": True},
                )
                messages.insert(-1, new_request)

        return messages
