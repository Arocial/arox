import logging
from dataclasses import dataclass

from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import CommandEvent, Plugin, command
from arox.plugins.capabilities import AGENT_INFO, AGENT_RESET, SUBAGENT

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SetModelEvent(CommandEvent):
    model_ref: str


@dataclass(kw_only=True)
class InfoEvent(CommandEvent):
    pass


@dataclass(kw_only=True)
class ResetEvent(CommandEvent):
    pass


@dataclass(kw_only=True)
class AgentCallEvent(CommandEvent):
    subagent_name: str
    task: str


class CorePlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)

        cm = self.agent.command_manager
        cm.register_handler(SetModelEvent, self._handle_set_model_event)
        cm.register_handler(InfoEvent, self._handle_info_event)
        cm.register_handler(ResetEvent, self._handle_reset_event)
        cm.register_handler(AgentCallEvent, self._handle_agent_call_event)

    async def _handle_set_model_event(self, event: SetModelEvent) -> str:
        if not event.model_ref:
            return "Please specify a model name"
        self.agent.set_model(event.model_ref)
        return f"Model switched to {event.model_ref}"

    async def _handle_info_event(self, event) -> str:
        lines = [f"Current model: {getattr(self.agent, 'provider_model', 'Unknown')}"]
        for provider in self.agent.get_capability(AGENT_INFO):
            info = await provider()
            if info:
                lines.append(info)
        return "\n".join(lines)

    async def _handle_reset_event(self, event) -> str:
        self.agent.reset()
        for provider in self.agent.get_capability(AGENT_RESET):
            provider()
        return "Reset complete."

    async def _handle_agent_call_event(self, event) -> str | None:
        if not event.subagent_name:
            return "Usage: /agent <name> [task]"

        subagent = None
        for get_subagent_func in self.agent.get_capability(SUBAGENT):
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

    @command("model", "Switch LLM model - /model <model_name>")
    def model_command(self, name: str, arg: str | None) -> CommandEvent:
        return SetModelEvent(model_ref=arg or "")

    @command("info", "Show current chat files and model in use - /info")
    def info_command(self, name: str, arg: str | None) -> CommandEvent:
        return InfoEvent()

    @command("reset", "Reset chat history and chat files - /reset")
    def reset_command(self, name: str, arg: str | None) -> CommandEvent:
        return ResetEvent()

    @command("agent", "Call a subagent - /agent <name> [task]")
    def agent_command(self, name: str, arg: str | None) -> CommandEvent:
        parts = arg.split(maxsplit=1) if arg else []
        return AgentCallEvent(
            subagent_name=parts[0] if parts else "",
            task=parts[1] if len(parts) > 1 else "",
        )
