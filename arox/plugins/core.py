import logging
from dataclasses import dataclass

from arox.core.io import RequestEvent
from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import Plugin, command
from arox.plugins.capabilities import AGENT_INFO, AGENT_RESET, SUBAGENT

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SetModelEvent(RequestEvent):
    model_ref: str


@dataclass(kw_only=True)
class InfoEvent(RequestEvent):
    pass


@dataclass(kw_only=True)
class ResetEvent(RequestEvent):
    pass


@dataclass(kw_only=True)
class AgentCallEvent(RequestEvent):
    subagent_name: str
    task: str


class CorePlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)

        self.agent.register_request_handler(SetModelEvent, self._handle_set_model_event)
        self.agent.register_request_handler(InfoEvent, self._handle_info_event)
        self.agent.register_request_handler(ResetEvent, self._handle_reset_event)
        self.agent.register_request_handler(
            AgentCallEvent, self._handle_agent_call_event
        )

    async def _handle_set_model_event(self, event: SetModelEvent) -> None:
        self.agent.set_model(event.model_ref)

    async def _handle_info_event(self, event) -> None:
        # Show current model
        current_model = getattr(self.agent, "provider_model", "Unknown")
        await self.agent.agent_io.send(f"Current model: {current_model}")

        for provider in self.agent.get_capability(AGENT_INFO):
            info = await provider()
            if info:
                await self.agent.agent_io.send(info)

    async def _handle_reset_event(self, event) -> None:
        self.agent.reset()
        for provider in self.agent.get_capability(AGENT_RESET):
            provider()
        await self.agent.agent_io.send("Reset complete.")

    async def _handle_agent_call_event(self, event) -> None:
        subagent = None
        for get_subagent_func in self.agent.get_capability(SUBAGENT):
            subagent = get_subagent_func(event.subagent_name)
            if subagent:
                break

        if not isinstance(subagent, DelegatableAgent):
            await self.agent.agent_io.send(
                f"Subagent '{event.subagent_name}' not found."
            )
            return

        self.agent.agent_session.add_event(
            "subagent_call",
            {"subagent": subagent.name, "task": event.task},
        )
        result = await subagent.run_task(event.task)
        if result:
            await self.agent.agent_io.send(result)

    @command("model", "Switch LLM model - /model <model_name>")
    async def model_command(self, name: str, arg: str):
        if not arg:
            await self.agent.agent_io.send("Please specify a model name")
            return
        await self.agent.adapter_io.send(SetModelEvent(model_ref=arg))
        await self.agent.agent_io.send(f"Model switched to {arg}")

    @command("info", "Show current chat files and model in use - /info")
    async def info_command(self, name: str, arg: str):
        await self.agent.adapter_io.send(InfoEvent())

    @command("reset", "Reset chat history and chat files - /reset")
    async def reset_command(self, name: str, arg: str):
        await self.agent.adapter_io.send(ResetEvent())

    @command("agent", "Call a subagent - /agent <name> [task]")
    async def agent_command(self, name: str, arg: str):
        parts = arg.split(maxsplit=1) if arg else []
        if not parts:
            await self.agent.agent_io.send("Usage: /agent <name> [task]")
            return

        subagent_name = parts[0]
        task = parts[1] if len(parts) > 1 else ""
        await self.agent.adapter_io.send(
            AgentCallEvent(subagent_name=subagent_name, task=task)
        )
