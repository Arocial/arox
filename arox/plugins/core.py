import logging

from arox.core.plugin import Plugin, command
from arox.plugins.capabilities import AGENT_INFO, AGENT_RESET, SUBAGENT

logger = logging.getLogger(__name__)


class CorePlugin(Plugin):
    @command("model", "Switch LLM model - /model <model_name>")
    async def model_command(self, name: str, arg: str):
        if not arg:
            await self.agent.agent_io.agent_send("Please specify a model name")
            return
        self.agent.set_model(arg)
        await self.agent.agent_io.agent_send(f"Model switched to {arg}")

    @command("info", "Show current chat files and model in use - /info")
    async def info_command(self, name: str, arg: str):
        # Show current model
        current_model = getattr(self.agent, "provider_model", "Unknown")
        await self.agent.agent_io.agent_send(f"Current model: {current_model}")

        for provider in self.agent.get_capability(AGENT_INFO):
            info = await provider()
            if info:
                await self.agent.agent_io.agent_send(info)

    @command("reset", "Reset chat history and chat files - /reset")
    async def reset_command(self, name: str, arg: str):
        self.agent.reset()
        for provider in self.agent.get_capability(AGENT_RESET):
            provider()
        await self.agent.agent_io.agent_send("Reset complete.")

    @command("agent", "Call a subagent - /agent <name> [task]")
    async def agent_command(self, name: str, arg: str):
        parts = arg.split(maxsplit=1)
        if not parts:
            await self.agent.agent_io.agent_send("Usage: /agent <name> [task]")
            return

        subagent_name = parts[0]
        task = parts[1] if len(parts) > 1 else ""

        subagent = None
        for get_subagent_func in self.agent.get_capability(SUBAGENT):
            subagent = get_subagent_func(subagent_name)
            if subagent:
                break

        if not subagent or getattr(subagent.agent_config, "internal", False):
            await self.agent.agent_io.agent_send(
                f"Subagent '{subagent_name}' not found."
            )
            return

        if hasattr(subagent, "handle_task"):
            result = await subagent.handle_task(task, main_agent=self.agent)
            if result:
                await self.agent.agent_io.agent_send(result)
        else:
            await self.agent.agent_io.agent_send(
                f"Subagent '{subagent_name}' does not support tasks."
            )
