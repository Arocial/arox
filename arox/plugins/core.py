import logging
from dataclasses import dataclass
from typing import ClassVar

from arox.core.completion import CompletionItem, CompletionRequest
from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin
from arox.plugins.slots import AGENT_INFO, AGENT_RESET, SUBAGENT

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
class AgentCallEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("agent",)
    description: ClassVar[str] = "Call a subagent - /agent <name> [task]"

    subagent_name: str
    task: str

    @classmethod
    def from_slash(cls, name, arg):
        parts = arg.split(maxsplit=1) if arg else []
        return cls(
            subagent_name=parts[0] if parts else "",
            task=parts[1] if len(parts) > 1 else "",
        )


class CorePlugin(Plugin):
    def commands(self):
        return [
            CommandSpec(SetModelEvent, self.handle_set_model, self.complete_model_ref),
            CommandSpec(InfoEvent, self.handle_info),
            CommandSpec(ResetEvent, self.handle_reset),
            CommandSpec(AgentCallEvent, self.handle_agent_call),
        ]

    async def handle_set_model(self, event: SetModelEvent) -> str:
        if not event.model_ref:
            return "Please specify a model name"
        self.agent.set_model(event.model_ref)
        return f"Model switched to {event.model_ref}"

    def complete_model_ref(self, req: CompletionRequest):
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
        for provider in self.agent.get_slot(AGENT_INFO):
            info = await provider()
            if info:
                lines.append(info)
        return "\n".join(lines)

    async def handle_reset(self, event: ResetEvent) -> str:
        self.agent.reset()
        for provider in self.agent.get_slot(AGENT_RESET):
            provider()
        return "Reset complete."

    async def handle_agent_call(self, event: AgentCallEvent) -> str | None:
        if not event.subagent_name:
            return "Usage: /agent <name> [task]"

        subagent = None
        for get_subagent_func in self.agent.get_slot(SUBAGENT):
            subagent = get_subagent_func(event.subagent_name)
            if subagent:
                break

        if not isinstance(subagent, DelegatableAgent):
            return f"Subagent '{event.subagent_name}' not found."

        self.agent.agent_session.add_event(
            "subagent_call",
            {"subagent": subagent.name, "task": event.task},
        )
        return await subagent.run_task(event.task)
