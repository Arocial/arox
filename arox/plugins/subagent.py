import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from arox.core.llm_base import (
    AgentStatus,
    LLMBaseAgent,
)
from arox.core.plugin import Plugin, tool
from arox.core.session import AgentRunInfo, AgentSession, SessionStatus
from arox.plugins.slots import RUN_SUBAGENT, SUBAGENTS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SubagentMode(StrEnum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


# Backwards compatibility aliases
SubagentTask = AgentSession
SubagentTaskStatus = SessionStatus


class SubagentPlugin(Plugin):
    """Manage resumable child-agent tasks for an agent."""

    def __init__(self, agent):
        super().__init__(agent)
        self.mode = SubagentMode.SIMPLE
        self.task_sessions: dict[str, AgentSession] = {}
        self._task_ids_by_name: dict[str, str] = {}
        self._task_ids_by_target: dict[str, str] = {}
        self._active_delegations: dict[str, LLMBaseAgent] = {}
        self._lock = asyncio.Lock()

        def get_subagents(status: AgentStatus | str | None = None):
            active = []
            for session in self.task_sessions.values():
                if session.runtime is not None:
                    if (
                        status is None
                        or (
                            status == "active"
                            and session.runtime.status != AgentStatus.STOPPED
                        )
                        or session.runtime.status == status
                    ):
                        active.append(session.runtime)
            for subagent in self._active_delegations.values():
                if subagent not in active:
                    if (
                        status is None
                        or (
                            status == "active"
                            and subagent.status != AgentStatus.STOPPED
                        )
                        or subagent.status == status
                    ):
                        active.append(subagent)
            return active

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
                + "\nAvailable agent types:\n"
                + "\n".join(descriptions)
            )

        self.agent.provide_slot(SUBAGENTS, get_subagents)
        self.agent.provide_slot(SYSTEM_PROMPT, get_subagent_instructions)
        self.agent.provide_slot(RUN_SUBAGENT, self._delegate_once)

    def configure(self, config: dict[str, Any]) -> None:
        try:
            self.mode = SubagentMode(config.get("mode", SubagentMode.SIMPLE))
        except ValueError as exc:
            raise ValueError("subagent mode must be 'simple' or 'advanced'") from exc
        super().configure({**config, "mode": self.mode.value})

    def _simple_tools_enabled(self) -> bool:
        return self.mode is SubagentMode.SIMPLE

    def _advanced_tools_enabled(self) -> bool:
        return self.mode is SubagentMode.ADVANCED

    @staticmethod
    def _get_task_status(task_session: AgentSession) -> str:
        if task_session.status == SessionStatus.CLOSED:
            return "closed"
        if (
            task_session.running_task is not None
            and not task_session.running_task.done()
        ) or (
            task_session.runtime is not None
            and task_session.runtime.status == AgentStatus.RUNNING
        ):
            return "running"
        if task_session.last_error:
            if "interrupt" in task_session.last_error.lower():
                return "interrupted"
            return "error"
        if task_session.last_result is not None:
            return "completed"
        if task_session.running_task is not None:
            return "pending"
        return "idle"

    async def on_start(self) -> None:
        main_session = self.agent.session
        session_manager = main_session.manager if main_session else None
        if main_session is None or session_manager is None:
            return

        for child_id in list(main_session.children):
            child_session = await session_manager.load_session(child_id, main_session)
            if not isinstance(child_session, AgentSession):
                continue

            if child_session.task_name is None and child_session.target is None:
                continue

            if (
                child_session.status == SessionStatus.ACTIVE
                and child_session.last_result is None
                and child_session.last_error is None
            ):
                child_session.record_interrupted(
                    "Parent process stopped before the task completed."
                )
                child_session.runtime = None
                child_session.running_task = None
                await child_session.save()

            if child_session.id in self.task_sessions or (
                child_session.task_name
                and child_session.task_name in self._task_ids_by_name
            ):
                logger.warning(
                    "Ignoring duplicate restored subagent task %s", child_session.target
                )
                continue
            self._register_task_session(child_session)

    async def on_stop(self) -> None:
        running = [
            session.running_task
            for session in self.task_sessions.values()
            if session.running_task and not session.running_task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    def _register_task_session(self, task_session: AgentSession) -> None:
        self.task_sessions[task_session.id] = task_session
        if task_session.task_name:
            self._task_ids_by_name[task_session.task_name] = task_session.id
        if task_session.target:
            self._task_ids_by_target[task_session.target] = task_session.id

    def _unregister_task_session(self, task_session: AgentSession) -> None:
        self.task_sessions.pop(task_session.id, None)
        if task_session.task_name:
            self._task_ids_by_name.pop(task_session.task_name, None)
        if task_session.target:
            self._task_ids_by_target.pop(task_session.target, None)

    def _validate_agent_type(self, agent_type: str) -> None:
        if agent_type not in self.agent.agent_config.subagents:
            raise ValueError(f"Agent type '{agent_type}' is not configured.")

    def _create_child_session(
        self,
        agent_type: str,
        *,
        task_name: str | None = None,
        message: str | None = None,
        status: SessionStatus = SessionStatus.ACTIVE,
    ) -> AgentSession:
        agent_config = self.agent.config.agent.get(agent_type)
        configured_type = agent_config.type if agent_config else "chat"
        workspace = str(self.agent.workspace) if self.agent.workspace else None

        if self.agent.session:
            return self.agent.session.create_child_session(
                agent_name=agent_type,
                agent_type=configured_type,
                workspace=workspace,
                task_name=task_name,
                target=f"/{self.agent.name}/{task_name}" if task_name else None,
                initial_message=message,
                last_message=message,
                status=status,
            )

        return AgentSession(
            agent_name=agent_type,
            agent_type=configured_type,
            agent_source="static",
            workspace=workspace,
            task_name=task_name,
            target=f"/{self.agent.name}/{task_name}" if task_name else None,
            initial_message=message,
            last_message=message,
            status=status,
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
        )

    @staticmethod
    async def _notify_subagent_created(
        callback: Callable[[LLMBaseAgent], Awaitable[None] | None] | None,
        runtime: LLMBaseAgent,
    ) -> None:
        if callback is None:
            return
        result = callback(runtime)
        if asyncio.iscoroutine(result):
            await result

    async def _unlink_child_session(self, task_session: AgentSession) -> None:
        if self.agent.session and task_session.id in self.agent.session.children:
            self.agent.session.children.remove(task_session.id)
            await self.agent.session.save()

    def _create_task_runtime(self, task_session: AgentSession) -> LLMBaseAgent:
        return task_session.create_agent(
            config_loader=self.agent.config_loader,
            io_adapter=self.agent.io_adapter,
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

    def _running_task_count(self) -> int:
        return sum(
            1
            for session in self.task_sessions.values()
            if session.running_task is not None and not session.running_task.done()
        )

    async def _spawn_task(
        self,
        task_name: str,
        message: str,
        agent_type: str,
        on_subagent_created: Callable[[LLMBaseAgent], Awaitable[None] | None]
        | None = None,
    ) -> AgentSession:
        async with self._lock:
            if not _TASK_NAME_PATTERN.fullmatch(task_name):
                raise ValueError(
                    "task_name must start with a lowercase letter and contain only "
                    "lowercase letters, digits, or underscores (maximum 64 characters)."
                )
            if task_name in self._task_ids_by_name:
                raise ValueError(
                    f"Task '{task_name}' already exists. Use followup_task to continue it."
                )
            self._validate_agent_type(agent_type)
            if (
                self._running_task_count()
                >= self.agent.agent_config.max_parallel_subagents
            ):
                raise ValueError(
                    "Maximum parallel subagents reached: "
                    f"{self.agent.agent_config.max_parallel_subagents}."
                )

            task_session = self._create_child_session(
                agent_type,
                task_name=task_name,
                message=message,
                status=SessionStatus.ACTIVE,
            )
            self._register_task_session(task_session)

            try:
                runtime = self._create_task_runtime(task_session)
                await self._notify_subagent_created(on_subagent_created, runtime)
                await task_session.save()
                if self.agent.session:
                    await self.agent.session.save()
            except BaseException:
                self._unregister_task_session(task_session)
                await self._unlink_child_session(task_session)
                raise

            task_session.running_task = asyncio.create_task(
                self._execute_task(task_session, message, initial_runtime=runtime),
                name=f"subagent:{task_session.target}",
            )
            return task_session

    async def _execute_task(
        self,
        task_session: AgentSession,
        message: str,
        initial_runtime: LLMBaseAgent | None = None,
    ) -> str | None:
        task_session.last_message = message
        task_session.last_result = None
        task_session.last_error = None
        await task_session.save()
        await self.agent.broadcast_agent_info()

        runtime = initial_runtime or self._create_task_runtime(task_session)

        self.agent.session.record_subagent_call(task_session.agent_name, message)
        try:
            await self.agent.broadcast_agent_info()
            async with runtime:
                result = await runtime.step(message)
            output = result.output if isinstance(result.output, str) else None
            task_session.record_result(output)
            return output
        except asyncio.CancelledError:
            task_session.record_interrupted()
            raise
        except Exception as exc:
            task_session.record_error(exc)
            logger.exception("Subagent task %s failed", task_session.target)
            return None
        finally:
            task_session.running_task = None
            await task_session.save()
            await self.agent.broadcast_agent_info()

    async def _start_followup_turn(
        self, task_session: AgentSession, message: str
    ) -> None:
        async with self._lock:
            if task_session.running_task and not task_session.running_task.done():
                raise ValueError(
                    f"Agent task '{task_session.target}' is already running."
                )
            if (
                self._running_task_count()
                >= self.agent.agent_config.max_parallel_subagents
            ):
                raise ValueError(
                    "Maximum parallel subagents reached: "
                    f"{self.agent.agent_config.max_parallel_subagents}."
                )

            task_session.last_message = message
            task_session.last_result = None
            task_session.last_error = None
            await task_session.save()
            task_session.running_task = asyncio.create_task(
                self._execute_task(task_session, message),
                name=f"subagent:{task_session.target}:followup",
            )

    def _format_task_status(
        self, task_session: AgentSession, include_result: bool = True
    ) -> str:
        lines = [
            f"- task_id: {task_session.id}",
            f"- target: {task_session.target}",
            f"- agent_type: {task_session.agent_name}",
            f"- status: {self._get_task_status(task_session)}",
        ]
        if task_session.last_error:
            lines.append(f"- error: {task_session.last_error}")
        if include_result and task_session.last_result:
            lines.extend(("", "Result:", task_session.last_result))
        return "\n".join(lines)

    async def _wait_for_task(
        self, task_session: AgentSession, timeout_seconds: float | None
    ) -> str:
        running_task = task_session.running_task
        if running_task is not None:
            try:
                if timeout_seconds is None:
                    await asyncio.shield(running_task)
                else:
                    await asyncio.wait_for(
                        asyncio.shield(running_task), timeout=timeout_seconds
                    )
            except TimeoutError:
                return "Agent is still running.\n" + self._format_task_status(
                    task_session, False
                )
            except asyncio.CancelledError:
                if task_session.last_error != "Task interrupted.":
                    raise

        return self._format_task_status(task_session)

    async def _delegate_once(
        self,
        subagent_name: str,
        task: str,
        on_subagent_created: Callable[[LLMBaseAgent], Awaitable[None] | None]
        | None = None,
    ) -> str:
        self._validate_agent_type(subagent_name)
        task_session = self._create_child_session(subagent_name)

        try:
            runtime = self._create_task_runtime(task_session)
        except BaseException:
            await self._unlink_child_session(task_session)
            raise

        self._active_delegations[runtime.uuid] = runtime
        if self.agent.session:
            await self.agent.session.save()
        await self.agent.broadcast_agent_info()

        try:
            await self._notify_subagent_created(on_subagent_created, runtime)
            result = await self._execute_task(task_session, task, runtime)
            return result or "Task completed with no output."
        finally:
            self._active_delegations.pop(runtime.uuid, None)
            task_session.close_session()
            await task_session.save()
            await self.agent.broadcast_agent_info()

    @tool(sequential=True, enabled=_simple_tools_enabled)
    async def delegate_task(self, message: str, agent_type: str) -> str:
        """Delegate a task to a configured subagent and wait for its result."""
        return await self._delegate_once(agent_type, message)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def spawn_agent(self, task_name: str, message: str, agent_type: str) -> str:
        """Start a resumable task using a configured subagent.

        `task_name` must be a unique lowercase identifier and `message` must
        contain all context the subagent needs.
        """
        task_session = await self._spawn_task(task_name, message, agent_type)
        return "Agent spawned.\n" + self._format_task_status(task_session, False)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def followup_task(self, target: str, message: str) -> str:
        """Continue a completed, interrupted, or errored agent task.

        The existing agent session and message history are reused.
        """
        task_session = self._resolve_task(target)
        await self._start_followup_turn(task_session, message)
        return "Follow-up started.\n" + self._format_task_status(task_session, False)

    @tool(enabled=_advanced_tools_enabled)
    async def wait_agent(self, target: str, timeout_seconds: float = 60) -> str:
        """Wait for an agent task and return its latest result.

        Timing out does not interrupt the task.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        return await self._wait_for_task(self._resolve_task(target), timeout_seconds)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def interrupt_agent(self, target: str) -> str:
        """Interrupt a running agent turn while preserving its session for follow-up."""
        task_session = self._resolve_task(target)
        running_task = task_session.running_task
        if running_task is None or running_task.done():
            return "Agent is not running.\n" + self._format_task_status(
                task_session, False
            )

        running_task.cancel()
        await asyncio.gather(running_task, return_exceptions=True)
        return "Agent interrupted.\n" + self._format_task_status(task_session, False)

    @tool(enabled=_advanced_tools_enabled)
    async def list_agents(self, status: str | None = None) -> str:
        """List spawned agent tasks, optionally filtered by task status."""
        filter_status = (
            status.value
            if hasattr(status, "value")
            else (str(status).lower() if status is not None else None)
        )
        task_sessions = [
            task_session
            for task_session in self.task_sessions.values()
            if filter_status is None
            or self._get_task_status(task_session) == filter_status
        ]
        if not task_sessions:
            return "No agent tasks."

        blocks = []
        for task_session in sorted(task_sessions, key=lambda item: item.created_at):
            block = self._format_task_status(task_session, False)
            if task_session.last_result:
                summary = task_session.last_result.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
