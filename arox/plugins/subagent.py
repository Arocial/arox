import json
import logging
import uuid
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from arox.core.config import AgentConfig
from arox.core.llm_base import AgentInfoUpdate, DelegatableAgent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, ToolDef
from arox.core.session import AgentRunInfo, AgentSession
from arox.plugins.slots import (
    SUBAGENTS,
)
from arox.utils import import_class

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SubagentEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("subagent",)
    description: ClassVar[str] = (
        "Manage or call subagents - /subagent list|create|delete|call ..."
    )

    action: str = ""
    name: str = ""
    task: str = ""
    type: str | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_slash(cls, name, arg):
        parts = arg.split(maxsplit=3) if arg else []
        action = parts[0] if parts else ""
        name_arg = parts[1] if len(parts) > 1 else ""

        if action == "call":
            # For /subagent call <name> [task]
            # Re-split to get the full task
            sub_parts = arg.split(maxsplit=2)
            return cls(
                action="call",
                name=name_arg,
                task=sub_parts[2] if len(sub_parts) > 2 else "",
            )

        config = None
        if len(parts) > 3:
            config = json.loads(parts[3])
            if not isinstance(config, dict):
                raise ValueError("json-config must be an object")
        return cls(
            action=action,
            name=name_arg,
            type=parts[2] if len(parts) > 2 else None,
            config=config,
        )


class SubagentPlugin(Plugin):
    """Manages subagents and exposes them to the main agent.

    Subagents are surfaced two ways: the ``delegate_to_subagent`` tool the LLM
    can call, and the ``/agent`` slash command the human can type. Both reach
    the subagents through the :data:`SUBAGENTS` slot.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.subagents: dict[str, Any] = {}

        def get_subagents():
            return list(self.subagents.values())

        self.agent.provide_slot(SUBAGENTS, get_subagents)

    async def _create_subagent(
        self,
        name: str,
        agent_config: AgentConfig,
        agent_source: Literal["static", "dynamic"],
    ):
        try:
            agent_cls = import_class(agent_config.type, group="arox.agents")
        except ValueError:
            raise ValueError(
                f"Unknown agent type: {agent_config.type} for agent {name}"
            )

        owner_session = self.agent.session

        sub_session = AgentSession(
            path=[*owner_session.path, str(uuid.uuid4())]
            if owner_session
            else [str(uuid.uuid4())],
            agent_name=name,
            agent_config=agent_config.model_copy(deep=True),
            agent_source=agent_source,
            workspace=str(self.agent.workspace),
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
        )
        sub_session.owner = owner_session
        sub_session.manager = owner_session.manager if owner_session else None
        if owner_session:
            owner_session.children.append(sub_session.id)

        subagent = agent_cls(
            self.agent.parsed_config,
            io_adapter=self.agent.io_adapter,
            workspace=self.agent.workspace,
            session=sub_session,
        )
        return subagent

    async def on_start(self):
        main_session = self.agent.session
        session_manager = main_session.manager if main_session else None
        if main_session is None or session_manager is None:
            return

        # 1. Reconstruct from child session ids.
        for child_id in main_session.children:
            subagent = await session_manager.build_from_session(
                child_id,
                main_session,
                parsed_config=self.agent.parsed_config,
                io_adapter=self.agent.io_adapter,
            )
            if subagent:
                self.subagents[subagent.name] = subagent
                self.agent.parsed_config.agent[subagent.name] = subagent.agent_config

        # 2. Add static agents not already in children
        started_names = set(self.subagents.keys())
        parsed_config = self.agent.parsed_config
        for agent_name in self.agent.agent_config.subagents:
            if agent_name not in started_names:
                agent_config = parsed_config.agent.get(agent_name)
                if not agent_config:
                    raise ValueError(f"Agent config for '{agent_name}' not found")
                subagent = await self._create_subagent(
                    name=agent_name,
                    agent_config=agent_config,
                    agent_source="static",
                )
                self.subagents[subagent.name] = subagent

        for subagent in self.subagents.values():
            await self.agent._stack.enter_async_context(subagent)
        await self._broadcast_agent_info()

    async def create_subagent(
        self,
        name: str,
        agent_type: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create a dynamic subagent that will be restored with the session."""
        if not name:
            raise ValueError("subagent name is required")
        if name in self.subagents:
            raise ValueError(f"Subagent '{name}' already exists")

        base = self.agent.parsed_config.agent.get(name)
        base_config = base.model_dump(mode="json") if base else {}
        merged_config_data = {**base_config, **(config or {})}
        if agent_type:
            merged_config_data["type"] = agent_type
        agent_config = AgentConfig.model_validate(merged_config_data)

        self.agent.parsed_config.agent[name] = agent_config

        subagent = await self._create_subagent(
            name=name,
            agent_config=agent_config,
            agent_source="dynamic",
        )
        self.subagents[subagent.name] = subagent
        await self.agent._stack.enter_async_context(subagent)
        self.agent.session.record_subagent_created(
            name, agent_config.model_dump(mode="json")
        )
        if self.agent.session.manager:
            await self.agent.session.manager.session_store.save_session(
                subagent.session
            )
        await self._broadcast_agent_info()
        return f"Created subagent '{name}'."

    async def _broadcast_agent_info(self):
        info = AgentInfoUpdate(agent_id=self.agent.uuid)
        await self.agent.agent_io.send(info)

    async def delete_subagent(self, name: str) -> str:
        """Delete a dynamic subagent from the current session."""
        if not name:
            raise ValueError("subagent name is required")

        subagent = self.subagents.pop(name, None)
        if subagent is None:
            raise ValueError(f"Dynamic subagent '{name}' not found")

        sub_session = subagent.session
        if sub_session.id in self.agent.session.children:
            self.agent.session.children.remove(sub_session.id)
        self.agent.session.record_subagent_deleted(
            name,
            sub_session.path,
        )
        if self.agent.session.manager:
            await self.agent.session.manager.session_store.delete_session(
                sub_session.path
            )
        await self._broadcast_agent_info()
        return f"Deleted subagent '{name}'."

    def commands(self):
        return [
            CommandSpec(SubagentEvent, self.handle_subagent_event),
        ]

    def tools(self):
        return [
            ToolDef(
                func=self.delegate_to_subagent,
                kwargs={"prepare": self._prepare_delegate},
            ),
            ToolDef(func=self.create_subagent),
        ]

    async def _delegatable_subagents(self) -> list:
        return [a for a in self.subagents.values() if isinstance(a, DelegatableAgent)]

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

        self.agent.session.record_subagent_call(agent.name, task)
        result = await agent.run_task(task)
        return result or "Task completed with no output."

    async def list_subagents(self) -> str:
        """List currently available subagents."""
        agents = list(self.subagents.values())
        if not agents:
            return "No subagents."
        lines = []
        for agent in sorted(agents, key=lambda a: a.name):
            source = agent.agent_source
            desc = agent.agent_config.description or "No description"
            lines.append(f"- {agent.name} ({source}): {desc}")
        return "\n".join(lines)

    async def handle_subagent_event(self, event: SubagentEvent) -> str | None:
        if event.action == "list":
            return await self.list_subagents()
        if event.action == "create" and event.name:
            return await self.create_subagent(event.name, event.type, event.config)
        if event.action == "delete" and event.name:
            return await self.delete_subagent(event.name)
        if event.action == "call":
            if not event.name:
                return "Usage: /subagent call <name> [task]"

            subagent = next(
                (a for a in self.subagents.values() if a.name == event.name),
                None,
            )

            if not isinstance(subagent, DelegatableAgent):
                return f"Subagent '{event.name}' not found."

            self.agent.session.record_subagent_call(subagent.name, event.task)
            return await subagent.run_task(event.task)

        return (
            "Usage:\n"
            "  /subagent list\n"
            "  /subagent create <name> [type] [json-config]\n"
            "  /subagent delete <name>\n"
            "  /subagent call <name> [task]"
        )
