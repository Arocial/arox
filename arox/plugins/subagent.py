import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from arox.core.llm_base import create_agent
from arox.core.plugin import CommandEvent, CommandSpec, Plugin, ToolDef
from arox.plugins.slots import (
    DELEGATE_TO_SUBAGENT,
    SUBAGENTS,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SubagentEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("subagent",)
    description: ClassVar[str] = "Manage or call subagents - /subagent list|call ..."

    action: str = ""
    name: str = ""
    task: str = ""

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

        return cls(
            action=action,
            name=name_arg,
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
        self.background_tasks: dict[str, asyncio.Task] = {}
        self.task_results: dict[str, str] = {}

        def get_subagents(status: str | None = None):
            if status is None:
                return list(self.subagents.values())
            return [s for s in self.subagents.values() if s.status == status]

        def get_subagent_instructions() -> str:
            subagent_names = self.agent.agent_config.subagents
            if not subagent_names:
                return ""

            descriptions = []
            for name in subagent_names:
                agent_config = self.agent.config.agent.get(name)
                desc = (
                    agent_config.description
                    if agent_config and agent_config.description
                    else "No description"
                )
                descriptions.append(f"- {name}: {desc}")

            return (
                "## Subagent Collaboration Framework\n"
                "You act as a Lead Orchestrator. You have access to specialized subagents.\n"
                "Available subagents:\n" + "\n".join(descriptions)
            )

        self.agent.provide_slot(SUBAGENTS, get_subagents)
        self.agent.provide_slot(SYSTEM_PROMPT, get_subagent_instructions)
        self.agent.provide_slot(DELEGATE_TO_SUBAGENT, self._internal_delegate)

    async def _create_subagent(
        self,
        name: str,
        agent_source: Literal["static", "dynamic"],
    ):
        subagent = create_agent(
            name=name,
            config_loader=self.agent.config_loader,
            io_adapter=self.agent.io_adapter,
            parent_session=self.agent.session,
            agent_source=agent_source,
            workspace=self.agent.workspace,
        )
        self.subagents[subagent.uuid] = subagent
        await self.agent.broadcast_agent_info()
        return subagent

    async def _destroy_subagent(self, subagent: Any):
        if hasattr(self.agent.io_adapter, "hosts"):
            self.agent.io_adapter.hosts.pop(subagent.uuid, None)

        subagent.close_session()

        self.agent.session.record_subagent_deleted(
            subagent.name,
            subagent.session.path,
        )
        if self.agent.session.manager:
            await self.agent.session.manager.save_session(subagent.session)
        await self.agent.broadcast_agent_info()

    async def on_start(self):
        main_session = self.agent.session
        session_manager = main_session.manager if main_session else None
        if main_session is None or session_manager is None:
            return

        for child_id in list(main_session.children):
            child_session = await session_manager.load_session(child_id, main_session)
            if child_session:
                subagent = create_agent(
                    name=child_session.agent_name,
                    config_loader=self.agent.config_loader,
                    io_adapter=self.agent.io_adapter,
                    session=child_session,
                )
                self.subagents[subagent.uuid] = subagent

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
            ToolDef(
                func=self.dispatch_background_task,
                kwargs={"prepare": self._prepare_delegate},
            ),
            ToolDef(func=self.check_task_status),
        ]

    async def _prepare_delegate(
        self, ctx: RunContext, tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        subagent_names = self.agent.agent_config.subagents
        if not subagent_names:
            # Hide the tool entirely when there is nothing to delegate to.
            return None
        return tool_def

    async def check_task_status(self, task_id: str) -> str:
        """Check the status or get the result of a background subagent task."""
        if task_id in self.task_results:
            result = self.task_results.pop(task_id)
            self.background_tasks.pop(task_id, None)
            return f"Task Completed. Result:\n{result}"

        if task_id in self.background_tasks:
            return f"Task {task_id} is still running."

        return f"Error: Unknown task ID '{task_id}'."

    async def dispatch_background_task(self, subagent_name: str, task: str) -> str:
        """Dispatch a long-running task to a subagent in the background and return a task_id.

        Use this for parallel or time-consuming tasks across different domains, then use `check_task_status` later.
        ALWAYS provide comprehensive context in your `task` description so the subagent doesn't lack necessary background.
        """
        full_task = task
        task_id = f"task_{uuid.uuid4().hex[:6]}"

        async def _run_and_store():
            subagent = await self._create_subagent(
                name=subagent_name,
                agent_source="static",
            )

            try:
                async with subagent:
                    self.agent.session.record_subagent_call(
                        subagent.name, f"[Background: {task_id}] " + full_task
                    )
                    res = await subagent.run_task(full_task)
                    self.task_results[task_id] = res or "Task completed with no output."
            except Exception as e:
                logger.error(f"Background task {task_id} failed", exc_info=True)
                self.task_results[task_id] = f"Failed with error: {str(e)}"
            finally:
                await self._destroy_subagent(subagent)

        coro = _run_and_store()
        self.background_tasks[task_id] = asyncio.create_task(coro)

        return f"Task dispatched to {subagent_name}. Task ID: {task_id}. Use check_task_status to get results later."

    async def _internal_delegate(
        self,
        subagent_name: str,
        task: str,
        on_subagent_created: Callable[[Any], Awaitable[None] | None] | None = None,
    ) -> str:
        subagent = await self._create_subagent(
            name=subagent_name,
            agent_source="static",
        )

        if on_subagent_created:
            res = on_subagent_created(subagent)
            if asyncio.iscoroutine(res):
                await res

        try:
            async with subagent:
                self.agent.session.record_subagent_call(subagent.name, task)
                result = await subagent.run_task(task)
                return result or "Task completed with no output."
        finally:
            await self._destroy_subagent(subagent)

    async def delegate_to_subagent(self, subagent_name: str, task: str) -> str:
        """Delegate a task to a specific subagent.

        Wait for the subagent to complete and return its result.
        Use this for sequential tasks.
        ALWAYS provide comprehensive context in your `task` description so the subagent doesn't lack necessary background.
        """
        return await self._internal_delegate(subagent_name, task)

    async def list_subagents(self) -> str:
        """List currently available subagents."""
        subagent_names = self.agent.agent_config.subagents
        if not subagent_names:
            return "No subagents."
        lines = []
        for name in sorted(subagent_names):
            agent_config = self.agent.config.agent.get(name)
            desc = (
                agent_config.description
                if agent_config and agent_config.description
                else "No description"
            )
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    async def handle_subagent_event(self, event: SubagentEvent) -> str | None:
        if event.action == "list":
            return await self.list_subagents()
        if event.action == "call":
            if not event.name:
                return "Usage: /subagent call <name> [task]"
            return await self.delegate_to_subagent(event.name, event.task)

        return "Usage:\n  /subagent list\n  /subagent call <name> [task]"
