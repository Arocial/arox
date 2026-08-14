import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from arox.core.agent_runtime import (
    AgentRuntime,
)
from arox.core.plugin import Plugin, tool
from arox.core.runner import TaskSessionRunner
from arox.core.session import AgentSession
from arox.plugins.slots import RUN_SUBAGENT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

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
        self.runtime.provide_slot(RUN_SUBAGENT, self._delegate_once)

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

    @staticmethod
    async def _close_runner(task_session: AgentSession) -> None:
        if task_session.runner is not None:
            await task_session.runner.stop()

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

    async def _discard_task_session(self, task_session: AgentSession) -> None:
        self._unregister_task_session(task_session)
        await self.runtime.session.manager.remove_child(
            self.runtime.session, task_session
        )

    def _resolve_task(self, target: str) -> AgentSession:
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
        on_subagent_created: Callable[[AgentRuntime], Awaitable[None] | None]
        | None = None,
    ) -> AgentSession:
        async with self._lock:
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
                last_message=message,
            )
            self._register_task_session(task_session)

            try:
                await self._start_session_task(
                    task_session, message, on_subagent_created
                )
            except BaseException:
                await self._close_runner(task_session)
                await self._discard_task_session(task_session)
                raise
            return task_session

    async def _ensure_runner(
        self,
        task_session: AgentSession,
        on_agent_created: Callable[[AgentRuntime], Awaitable[None] | None]
        | None = None,
    ) -> TaskSessionRunner:
        runner = task_session.runner
        if runner is None:
            runner = TaskSessionRunner(
                task_session, self.runtime.config_loader, self.runtime.io_adapter
            )
            runtime = await runner.start()
            if on_agent_created is not None:
                callback_result = on_agent_created(runtime)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
        return runner

    async def _start_session_task(
        self,
        task_session: AgentSession,
        message: str,
        on_agent_created: Callable[[AgentRuntime], Awaitable[None] | None]
        | None = None,
    ) -> asyncio.Task[Any]:
        self.runtime.session.record_subagent_call(task_session.agent_name, message)
        runner = await self._ensure_runner(task_session, on_agent_created)
        task = runner.start_turn(message)
        return task

    def _format_task(
        self, task_session: AgentSession, include_result: bool = True
    ) -> str:
        lines = [
            f"- task_id: {task_session.id}",
            f"- target: {task_session.target}",
            f"- agent_name: {task_session.agent_name}",
        ]
        if task_session.error:
            lines.append(f"- error: {task_session.error}")
        if include_result and task_session.result:
            lines.extend(("", "Result:", task_session.result))
        return "\n".join(lines)

    async def _delegate_once(
        self,
        subagent_name: str,
        task: str,
        on_subagent_created: Callable[[AgentRuntime], Awaitable[None] | None]
        | None = None,
    ) -> str:
        task_session = await self._spawn_task(
            None,
            task,
            subagent_name,
            on_subagent_created,
        )
        try:
            runner = task_session.runner
            if not isinstance(runner, TaskSessionRunner):
                raise TypeError("Subagent session is not using a TaskSessionRunner.")
            try:
                result = await runner.wait()
                if result is None or not isinstance(result.output, str):
                    return "Task completed with no output."
                return result.output
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Agent task %s failed", task_session.id)
                return "Task completed with no output."
        finally:
            await self._close_runner(task_session)
            self._unregister_task_session(task_session)

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.SIMPLE,
    )
    async def delegate_task(self, message: str, agent_name: str) -> str:
        """Delegate a task to a configured subagent and wait for its result."""
        return await self._delegate_once(agent_name, message)

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
        task_session = self._resolve_task(target)
        async with self._lock:
            runner = task_session.runner
            if runner is not None and runner.current_task is not None:
                raise ValueError(
                    f"Agent task '{task_session.target}' is already running."
                )

            await self._start_session_task(task_session, message)
        return "Follow-up started.\n" + self._format_task(task_session, False)

    @tool(enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED)
    async def wait_agent(self, target: str, timeout_seconds: float = 600) -> str:
        """Wait for an agent task and return its latest result.

        Timing out does not interrupt the task.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        task_session = self._resolve_task(target)
        runner = task_session.runner
        if runner is None:
            return self._format_task(task_session)
        try:
            await runner.wait(timeout_seconds)
        except TimeoutError:
            return "Agent is still running.\n" + self._format_task(task_session, False)

        return self._format_task(task_session)

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def interrupt_agent(self, target: str) -> str:
        """Interrupt a running agent turn while preserving its session for follow-up."""
        task_session = self._resolve_task(target)
        runner = task_session.runner
        if runner is None or not await runner.cancel_turn():
            return "Agent is not running.\n" + self._format_task(task_session, False)
        return "Agent interrupted.\n" + self._format_task(task_session, False)

    @tool(enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED)
    async def list_agents(self) -> str:
        """List spawned agent tasks."""
        task_sessions = list(self.task_sessions.values())
        if not task_sessions:
            return "No agent tasks."

        blocks = []
        for task_session in sorted(task_sessions, key=lambda item: item.created_at):
            block = self._format_task(task_session, False)
            if task_session.result:
                summary = task_session.result.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
