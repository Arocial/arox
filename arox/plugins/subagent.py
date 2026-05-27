import logging
from dataclasses import dataclass, replace
from typing import ClassVar

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, ToolDef
from arox.plugins.slots import DELEGATABLE_SUBAGENTS, SUBAGENT

logger = logging.getLogger(__name__)


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


class SubagentPlugin(Plugin):
    """Exposes the composer's subagents to the main agent.

    Subagents are surfaced two ways: the ``delegate_to_subagent`` tool the LLM
    can call, and the ``/agent`` slash command the human can type. Both reach
    the subagents through the :data:`SUBAGENT` / :data:`DELEGATABLE_SUBAGENTS`
    slots the composer provides.
    """

    def commands(self):
        return [CommandSpec(AgentCallEvent, self.handle_agent_call)]

    def tools(self):
        return [
            ToolDef(
                func=self.delegate_to_subagent,
                kwargs={"prepare": self._prepare_delegate},
            )
        ]

    def _delegatable_subagents(self) -> list:
        for provider in self.agent.get_slot(DELEGATABLE_SUBAGENTS):
            subagents = provider()
            if subagents:
                return subagents
        return []

    async def _prepare_delegate(
        self, ctx: RunContext, tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        subagents = self._delegatable_subagents()
        if not subagents:
            # Hide the tool entirely when there is nothing to delegate to.
            return None
        descriptions = "\n".join(
            f"- {agent.name}: {agent.agent_config.description or 'No description'}"
            for agent in subagents
        )
        return replace(
            tool_def,
            description=(
                "Delegate a task to a specific subagent.\n\n"
                f"Available subagents:\n{descriptions}"
            ),
        )

    async def delegate_to_subagent(self, subagent_name: str, task: str) -> str:
        """Delegate a task to a specific subagent."""
        subagents = {agent.name: agent for agent in self._delegatable_subagents()}
        agent = subagents.get(subagent_name)
        if not agent:
            return (
                f"Error: Subagent '{subagent_name}' not found. "
                f"Available subagents: {', '.join(subagents)}"
            )

        self.agent.agent_session.add_event(
            "subagent_call",
            {"subagent": agent.name, "task": task},
        )
        result = await agent.run_task(task)
        return result or "Task completed with no output."

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
