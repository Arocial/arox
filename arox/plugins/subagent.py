import logging
from dataclasses import dataclass, replace
from typing import ClassVar

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from arox.core.llm_base import DelegatableAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, ToolDef
from arox.plugins.slots import (
    AGENT_SESSION,
    RECORD_EVENT,
    SESSION_STORE,
    SET_SESSION,
    SUBAGENTS,
)
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
    the subagents through the :data:`SUBAGENTS` slot.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.subagents = {}

        # Instantiate subagents and publish the slot synchronously at
        # construction time. Doing this here (rather than in ``on_start``) means
        # the ``SUBAGENTS`` slot is available regardless of plugin start order —
        # e.g. ``SessionPlugin`` can restore subagent sessions even if it starts
        # before this plugin. Only entering each subagent's async context is
        # deferred to ``on_start``.
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

            self.subagents[agent_name] = subagent

        def list_subagents():
            return list(self.subagents.values())

        self.agent.provide_slot(SUBAGENTS, list_subagents)

    async def on_start(self):
        # Link each subagent's session under the main agent session: hand the
        # subagent its owner path and the shared session store via the SET_SESSION
        # slot. Its SessionPlugin then nests its session beneath the main one (via
        # ``owner_id``), so the main agent's recursive load resumes it next run.
        main_session = await self.agent.invoke_slot(AGENT_SESSION)
        session_store = await self.agent.invoke_slot(SESSION_STORE)
        # A configured session store with no session yet means the SessionPlugin
        # hasn't run its on_start: this plugin is ordered before it. on_start
        # runs in plugin-list order, so subagents would silently get no session.
        if session_store is not None and main_session is None:
            raise RuntimeError(
                "SubagentPlugin.on_start ran before the session was initialized; "
                "order the 'session' plugin before 'subagent' in the agent config."
            )
        # Subsessions already loaded under the main session (by load_session's
        # recursive load) are handed to their subagent as-is, so each subagent
        # adopts its session off the shared tree instead of re-loading it from
        # disk. A subagent without a loaded session gets a fresh one, which we
        # attach to the main session's children so /fork can find it.
        existing = (
            {c.agent_name: c for c in main_session.children}
            if main_session is not None
            else {}
        )
        for subagent in self.subagents.values():
            preset = existing.get(subagent.name)
            if main_session is not None:
                await subagent.invoke_slot(
                    SET_SESSION,
                    None,
                    [*main_session.owner_path, main_session.id],
                    session_store,
                    preset,
                )
            await self.agent._stack.enter_async_context(subagent)
            if main_session is not None and preset is None:
                sub_session = await subagent.invoke_slot(AGENT_SESSION)
                if sub_session is not None:
                    main_session.children.append(sub_session)

    def commands(self):
        return [CommandSpec(AgentCallEvent, self.handle_agent_call)]

    def tools(self):
        return [
            ToolDef(
                func=self.delegate_to_subagent,
                kwargs={"prepare": self._prepare_delegate},
            )
        ]

    async def _all_subagents(self) -> list:
        return await self.agent.invoke_slot(SUBAGENTS) or []

    async def _delegatable_subagents(self) -> list:
        return [
            a for a in await self._all_subagents() if isinstance(a, DelegatableAgent)
        ]

    async def _prepare_delegate(
        self, ctx: RunContext, tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        subagents = await self._delegatable_subagents()
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
        subagents = {agent.name: agent for agent in await self._delegatable_subagents()}
        agent = subagents.get(subagent_name)
        if not agent:
            return (
                f"Error: Subagent '{subagent_name}' not found. "
                f"Available subagents: {', '.join(subagents)}"
            )

        await self.agent.invoke_slot(
            RECORD_EVENT,
            "subagent_call",
            {"subagent": agent.name, "task": task},
        )
        result = await agent.run_task(task)
        return result or "Task completed with no output."

    async def handle_agent_call(self, event: AgentCallEvent) -> str | None:
        if not event.subagent_name:
            return "Usage: /agent <name> [task]"

        subagent = next(
            (a for a in await self._all_subagents() if a.name == event.subagent_name),
            None,
        )

        if not isinstance(subagent, DelegatableAgent):
            return f"Subagent '{event.subagent_name}' not found."

        await self.agent.invoke_slot(
            RECORD_EVENT,
            "subagent_call",
            {"subagent": subagent.name, "task": event.task},
        )
        return await subagent.run_task(event.task)
