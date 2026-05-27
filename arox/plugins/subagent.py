import logging
from dataclasses import dataclass, replace
from typing import ClassVar

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, ToolDef
from arox.plugins.slots import ALL_AGENTS, DELEGATABLE_SUBAGENTS, SUBAGENT
from arox.utils import import_class

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
    """Manages subagents and exposes them to the main agent.

    Subagents are surfaced two ways: the ``delegate_to_subagent`` tool the LLM
    can call, and the ``/agent`` slash command the human can type. Both reach
    the subagents through the :data:`SUBAGENT` / :data:`DELEGATABLE_SUBAGENTS`
    slots.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.subagents = {}

        # Instantiate subagents and publish their slots synchronously at
        # construction time. Doing this here (rather than in ``on_start``) means
        # the ``SUBAGENT`` / ``DELEGATABLE_SUBAGENTS`` / ``ALL_AGENTS`` slots are
        # available regardless of plugin start order — e.g. ``SessionPlugin``
        # can restore subagent sessions even if it starts before this plugin.
        # Only entering each subagent's async context is deferred to
        # ``on_start``.
        parsed_config = self.agent.parsed_config
        for agent_name in self.agent.agent_config.subagents:
            agent_config = parsed_config.agent.get(agent_name)
            if not agent_config:
                raise ValueError(f"Agent config for '{agent_name}' not found")

            agent_type = agent_config.type
            try:
                agent_cls = import_class(agent_type, group="arox.agents")
            except ValueError:
                raise ValueError(
                    f"Unknown agent type: {agent_type} for agent {agent_name}"
                )

            subagent = agent_cls(
                agent_name,
                parsed_config,
                io_adapter=self.agent.io_adapter,
                workspace=self.agent.workspace,
            )

            # Load hooks for subagent
            for hook_path in agent_config.pre_step_hooks:
                hook_func = import_class(hook_path, group="arox.hooks")
                subagent.add_pre_step_hook(hook_func)

            for hook_path in agent_config.post_step_hooks:
                hook_func = import_class(hook_path, group="arox.hooks")
                subagent.add_post_step_hook(hook_func)

            self.subagents[agent_name] = subagent

        def get_subagent(name: str):
            return self.subagents.get(name)

        self.agent.provide_slot(SUBAGENT, get_subagent)

        def list_delegatable_subagents():
            return [
                agent
                for agent in self.subagents.values()
                if isinstance(agent, DelegatableAgent)
            ]

        self.agent.provide_slot(DELEGATABLE_SUBAGENTS, list_delegatable_subagents)

        def get_all_subagents():
            return list(self.subagents.values())

        self.agent.provide_slot(ALL_AGENTS, get_all_subagents)

    async def on_start(self):
        for subagent in self.subagents.values():
            await self.agent._stack.enter_async_context(subagent)

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

        await self.agent.record_event(
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

        await self.agent.record_event(
            "subagent_call",
            {"subagent": subagent.name, "task": event.task},
        )
        return await subagent.run_task(event.task)
