import asyncio
import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from arox.core.agent_runtime import (
    AgentRuntime,
)
from arox.core.plugin import Plugin, tool
from arox.core.runner import TaskRunner
from arox.core.session import AgentSession
from arox.plugins.slots import SYSTEM_PROMPT

_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SubagentMode(StrEnum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


class SubagentPlugin(Plugin):
    """Manage resumable child-agent tasks for a runtime."""

    def __init__(self, runtime):
        super().__init__(runtime)
        self.mode = SubagentMode.SIMPLE
        self.task_sessions: dict[str, AgentSession] = {}
        self._task_ids_by_name: dict[str, str] = {}
        self._task_ids_by_target: dict[str, str] = {}
        self._lock = asyncio.Lock()

        def get_subagent_instructions() -> str:
            subagent_names = self.runtime.agent_config.subagents
            if not subagent_names:
                return ""

            descriptions = []
            for name in subagent_names:
                agent_config = self.runtime.config.agent.get(name)
                desc = (
                    agent_config.description
                    if agent_config and agent_config.description
                    else "No description"
                )
                descriptions.append(f"- {name}: {desc}")

            simple_prompt = (
                "You can delegate tasks synchronously to specialized agents. "
                "Use delegate_task to delegate a task. It runs synchronously and "
                "returns the completed result before you continue."
            )
            advanced_prompt = (
                "You can run specialized agents as resumable tasks. "
                "Use spawn_agent for independent background work, wait_agent "
                "to collect results, followup_task to continue an existing task, "
                "interrupt_agent to stop a running turn while preserving its "
                "context, and list_agents to inspect task state."
            )
            prompt = (
                simple_prompt if self.mode is SubagentMode.SIMPLE else advanced_prompt
            )
            return (
                "## Subagent Collaboration Framework\n"
                + prompt
                + "\nAvailable agents:\n"
                + "\n".join(descriptions)
            )

        self.runtime.provide_slot(SYSTEM_PROMPT, get_subagent_instructions)

    def configure(self, config: dict[str, Any]) -> None:
        try:
            self.mode = SubagentMode(config.get("mode", SubagentMode.SIMPLE))
        except ValueError as exc:
            raise ValueError("subagent mode must be 'simple' or 'advanced'") from exc
        super().configure({**config, "mode": self.mode.value})

    async def on_start(self) -> None:
        main_session = self.runtime.session
        session_manager = main_session.manager

        for child_session in await session_manager.children_of(main_session):
            if child_session.agent_source == "subagent":
                self._register_task_session(child_session)

    async def on_stop(self) -> None:
        session_manager = self.runtime.session.manager
        if session_manager is not None:
            await session_manager.stop_descendants(self.runtime.session)

    def _register_task_session(self, task_session: AgentSession) -> None:
        self.task_sessions[task_session.id] = task_session
        if task_session.task_name:
            self._task_ids_by_name[task_session.task_name] = task_session.id
        if task_session.target:
            self._task_ids_by_target[task_session.target] = task_session.id

    def _unregister_task_session(self, task_session: AgentSession) -> None:
        self.task_sessions.pop(task_session.id, None)
        if (
            task_session.task_name
            and self._task_ids_by_name.get(task_session.task_name) == task_session.id
        ):
            self._task_ids_by_name.pop(task_session.task_name, None)
        if (
            task_session.target
            and self._task_ids_by_target.get(task_session.target) == task_session.id
        ):
            self._task_ids_by_target.pop(task_session.target, None)

    def _resolve_session(self, target: str) -> AgentSession:
        task_id = target if target in self.task_sessions else None
        if task_id is None:
            task_id = self._task_ids_by_target.get(target)
        if task_id is None:
            task_id = self._task_ids_by_name.get(target)
        if task_id is None:
            raise ValueError(f"Unknown agent task '{target}'.")
        return self.task_sessions[task_id]

    async def _spawn_task(
        self,
        task_name: str | None,
        message: str,
        subagent_name: str,
    ) -> AgentSession:
        async with self._lock:
            # create session
            if task_name is not None and not _TASK_NAME_PATTERN.fullmatch(task_name):
                raise ValueError(
                    "task_name must start with a lowercase letter and contain only "
                    "lowercase letters, digits, or underscores (maximum 64 characters)."
                )
            if task_name is not None and task_name in self._task_ids_by_name:
                raise ValueError(
                    f"Task '{task_name}' already exists. Use followup_task to continue it."
                )
            if subagent_name not in self.runtime.agent_config.subagents:
                raise ValueError(f"Agent '{subagent_name}' is not configured.")

            task_session = await self.runtime.session.create_child_session(
                agent_name=subagent_name,
                agent_source="subagent",
                workspace=self.runtime.workspace,
                task_name=task_name,
                target=f"/{self.runtime.name}/{task_name}" if task_name else None,
                initial_message=message,
            )
            self._register_task_session(task_session)

            await self._start_task(task_session, message)
            return task_session

    async def _start_task(
        self,
        task_session: AgentSession,
        message: str,
    ) -> asyncio.Task[Any]:
        try:
            self.runtime.session.record_subagent_call(task_session.agent_name, message)
            runner = task_session.runner
            if runner is None:
                runner = TaskRunner(
                    task_session, self.runtime.config_loader, self.runtime.io_adapter
                )
                await runner.start_runtime()
            return runner.run(message)
        except BaseException:
            if runner is not None:
                await runner.stop_runtime()
            self._unregister_task_session(task_session)
            raise

    def _format_task(
        self, task_session: AgentSession, include_result: bool = True
    ) -> str:
        lines = [
            f"- task_id: {task_session.id}",
            f"- target: {task_session.target}",
            f"- agent_name: {task_session.agent_name}",
        ]
        runner = task_session.runner
        if isinstance(runner, TaskRunner):
            if runner.error:
                lines.append(f"- error: {runner.error}")
            if (
                include_result
                and runner.result
                and isinstance(runner.result.output, str)
            ):
                lines.extend(("", "Result:", runner.result.output))
        return "\n".join(lines)

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.SIMPLE,
    )
    async def delegate_task(self, message: str, agent_name: str) -> str:
        """Delegate a task to a configured subagent and wait for its result."""
        task_session = await self._spawn_task(None, message, agent_name)
        runner = task_session.runner
        try:
            if not isinstance(runner, TaskRunner) or runner.task is None:
                return "Task completed with no output."
            await asyncio.gather(runner.task, return_exceptions=True)
            result = runner.result
            if result is None or not isinstance(result.output, str):
                return "Task completed with no output."
            return result.output
        finally:
            await runner.stop_runtime()

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def spawn_agent(self, task_name: str, message: str, agent_name: str) -> str:
        """Start a resumable task using a configured subagent.

        `task_name` must be a unique lowercase identifier and `message` must
        contain all context the subagent needs.
        """
        task_session = await self._spawn_task(task_name, message, agent_name)
        return "Agent spawned.\n" + self._format_task(task_session, False)

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def followup_task(self, target: str, message: str) -> str:
        """Continue a completed, interrupted, or errored agent task.

        The existing agent session and message history are reused.
        """
        async with self._lock:
            task_session = self._resolve_session(target)
            runner = task_session.runner
            if (
                isinstance(runner, TaskRunner)
                and runner.task is not None
                and not runner.task.done()
            ):
                raise ValueError(
                    f"Agent task '{task_session.target}' is already running."
                )

            await self._start_task(task_session, message)
        return "Follow-up started.\n" + self._format_task(task_session, False)

    @tool(enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED)
    async def wait_agent(self, target: str, timeout_seconds: float = 600) -> str:
        """Wait for an agent task and return its latest result.

        Timing out does not interrupt the task.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        task_session = self._resolve_session(target)
        runner = task_session.runner

        if not isinstance(runner, TaskRunner) or runner.task is None:
            return self._format_task(task_session)
        done, _ = await asyncio.wait({runner.task}, timeout=timeout_seconds)
        if not done:
            return "Agent is still running.\n" + self._format_task(task_session, False)

        await asyncio.gather(runner.task, return_exceptions=True)
        outcome = self._format_task(task_session)
        await runner.stop_runtime()
        return outcome

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def interrupt_agent(self, target: str) -> str:
        """Interrupt a turn and release its runner while preserving its session."""
        task_session = self._resolve_session(target)
        async with self._lock:
            runner = task_session.runner
            if not isinstance(runner, TaskRunner):
                return "Agent is not running.\n" + self._format_task(
                    task_session, False
                )
            task = runner.task
            if task is None or task.done():
                return "Agent is not running.\n" + self._format_task(
                    task_session, False
                )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            outcome = self._format_task(task_session)
            await runner.stop_runtime()
        return "Agent interrupted.\n" + outcome

    @tool(enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED)
    async def list_agents(self) -> str:
        """List spawned agent tasks."""
        task_sessions = list(self.task_sessions.values())
        if not task_sessions:
            return "No agent tasks."

        blocks = []
        for task_session in sorted(task_sessions, key=lambda item: item.created_at):
            block = self._format_task(task_session, False)
            runner = task_session.runner
            if (
                isinstance(runner, TaskRunner)
                and runner.result
                and isinstance(runner.result.output, str)
            ):
                summary = runner.result.output.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
