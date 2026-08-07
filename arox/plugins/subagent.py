import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from arox.core.llm_base import (
    DelegatableAgent,
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
        self.tasks: dict[str, AgentSession] = {}
        self._tasks_by_name: dict[str, str] = {}
        self._tasks_by_target: dict[str, str] = {}
        self._active_run_subagents: dict[str, DelegatableAgent] = {}
        self._lock = asyncio.Lock()

        def get_subagents(status: str | None = None):
            active = []
            for session in self.tasks.values():
                if session.runtime is not None:
                    if status is None or session.status == status:
                        active.append(session.runtime)
            for subagent in self._active_run_subagents.values():
                if subagent not in active:
                    if status is None or subagent.status == status:
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
        self.agent.provide_slot(RUN_SUBAGENT, self._run_subagent_once)

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

            if child_session.status in (
                SessionStatus.PENDING,
                SessionStatus.RUNNING,
                SessionStatus.ACTIVE,
            ):
                child_session.record_interrupted(
                    "Parent process stopped before the task completed."
                )
                child_session.runtime = None
                child_session.running_task = None
                await child_session.save()

            if child_session.id in self.tasks or (
                child_session.task_name
                and child_session.task_name in self._tasks_by_name
            ):
                logger.warning(
                    "Ignoring duplicate restored subagent task %s", child_session.target
                )
                continue
            self._register_record(child_session)

    async def on_stop(self) -> None:
        running = [
            session.running_task
            for session in self.tasks.values()
            if session.running_task and not session.running_task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    def _register_record(self, session: AgentSession) -> None:
        self.tasks[session.id] = session
        if session.task_name:
            self._tasks_by_name[session.task_name] = session.id
        if session.target:
            self._tasks_by_target[session.target] = session.id

    @staticmethod
    def _as_delegatable(candidate: Any, agent_type: str) -> DelegatableAgent:
        if not isinstance(candidate, DelegatableAgent):
            raise TypeError(
                f"Agent type '{agent_type}' does not support delegated tasks."
            )
        return candidate

    async def _discard_new_session(self, session: AgentSession) -> None:
        if self.agent.session and session.id in self.agent.session.children:
            self.agent.session.children.remove(session.id)
            await self.agent.session.save()

    def _create_runtime(self, session: AgentSession) -> DelegatableAgent:
        candidate = session.create_agent(
            config_loader=self.agent.config_loader,
            io_adapter=self.agent.io_adapter,
        )
        return self._as_delegatable(candidate, session.agent_name)

    def _resolve_target(self, target: str) -> AgentSession:
        task_id = target if target in self.tasks else None
        if task_id is None:
            task_id = self._tasks_by_target.get(target)
        if task_id is None:
            task_id = self._tasks_by_name.get(target)
        if task_id is None:
            raise ValueError(f"Unknown agent task '{target}'.")
        return self.tasks[task_id]

    def _running_count(self) -> int:
        return sum(
            session.status
            in (SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.ACTIVE)
            for session in self.tasks.values()
        )

    async def _spawn_record(
        self,
        task_name: str,
        message: str,
        agent_type: str,
        on_subagent_created: Callable[[DelegatableAgent], Awaitable[None] | None]
        | None = None,
    ) -> AgentSession:
        async with self._lock:
            if not _TASK_NAME_PATTERN.fullmatch(task_name):
                raise ValueError(
                    "task_name must start with a lowercase letter and contain only "
                    "lowercase letters, digits, or underscores (maximum 64 characters)."
                )
            if task_name in self._tasks_by_name:
                raise ValueError(
                    f"Task '{task_name}' already exists. Use followup_task to continue it."
                )
            if agent_type not in self.agent.agent_config.subagents:
                raise ValueError(f"Agent type '{agent_type}' is not configured.")
            if self._running_count() >= self.agent.agent_config.max_parallel_subagents:
                raise ValueError(
                    "Maximum parallel subagents reached: "
                    f"{self.agent.agent_config.max_parallel_subagents}."
                )

            agent_config = self.agent.config.agent.get(agent_type)
            configured_type = agent_config.type if agent_config else "chat"

            if self.agent.session:
                child_session = self.agent.session.create_child_session(
                    agent_name=agent_type,
                    agent_type=configured_type,
                    workspace=str(self.agent.workspace)
                    if self.agent.workspace
                    else None,
                    task_name=task_name,
                    target=f"/{self.agent.name}/{task_name}",
                    initial_message=message,
                    last_message=message,
                    status=SessionStatus.PENDING,
                )
            else:
                child_session = AgentSession(
                    agent_name=agent_type,
                    agent_type=configured_type,
                    agent_source="static",
                    workspace=str(self.agent.workspace)
                    if self.agent.workspace
                    else None,
                    task_name=task_name,
                    target=f"/{self.agent.name}/{task_name}",
                    initial_message=message,
                    last_message=message,
                    status=SessionStatus.PENDING,
                    run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
                )

            self._register_record(child_session)

            try:
                subagent = self._create_runtime(child_session)
                if on_subagent_created:
                    result = on_subagent_created(subagent)
                    if asyncio.iscoroutine(result):
                        await result
                await child_session.save()
                if self.agent.session:
                    await self.agent.session.save()
            except BaseException:
                self.tasks.pop(child_session.id, None)
                if child_session.task_name:
                    self._tasks_by_name.pop(child_session.task_name, None)
                if child_session.target:
                    self._tasks_by_target.pop(child_session.target, None)
                await self._discard_new_session(child_session)
                raise

            child_session.running_task = asyncio.create_task(
                self._run_turn(child_session, message, initial_subagent=subagent),
                name=f"subagent:{child_session.target}",
            )
            return child_session

    async def _run_turn(
        self,
        session: AgentSession,
        message: str,
        initial_subagent: DelegatableAgent | None = None,
    ) -> str | None:
        session.status = SessionStatus.RUNNING
        session.last_message = message
        session.last_result = None
        session.last_error = None
        if self.agent.session:
            self.agent.session.record_subagent_call(session.agent_name, message)
        await session.save()
        await self.agent.broadcast_agent_info()

        subagent = initial_subagent or self._create_runtime(session)

        try:
            async with subagent:
                await self.agent.broadcast_agent_info()
                result = await subagent.run_task(message)
                session.record_result(result or "Task completed with no output.")
        except asyncio.CancelledError:
            session.record_interrupted("Task interrupted.")
            raise
        except Exception as exc:
            logger.exception("Subagent task %s failed", session.target)
            session.record_error(exc)
        finally:
            session.running_task = None
            await session.save()
            await self.agent.broadcast_agent_info()

        return session.last_result

    async def _start_followup(self, session: AgentSession, message: str) -> None:
        async with self._lock:
            if session.running_task and not session.running_task.done():
                raise ValueError(f"Agent task '{session.target}' is already running.")
            if self._running_count() >= self.agent.agent_config.max_parallel_subagents:
                raise ValueError(
                    "Maximum parallel subagents reached: "
                    f"{self.agent.agent_config.max_parallel_subagents}."
                )

            session.status = SessionStatus.PENDING
            session.last_message = message
            session.last_result = None
            session.last_error = None
            await session.save()
            session.running_task = asyncio.create_task(
                self._run_turn(session, message),
                name=f"subagent:{session.target}:followup",
            )

    def _format_status(self, session: AgentSession, include_result: bool = True) -> str:
        lines = [
            f"- task_id: {session.id}",
            f"- target: {session.target}",
            f"- agent_type: {session.agent_name}",
            f"- status: {session.status.value if isinstance(session.status, SessionStatus) else session.status}",
        ]
        if session.last_error:
            lines.append(f"- error: {session.last_error}")
        if include_result and session.last_result:
            lines.extend(("", "Result:", session.last_result))
        return "\n".join(lines)

    async def _wait_record(
        self, session: AgentSession, timeout_seconds: float | None
    ) -> str:
        running_task = session.running_task
        if running_task is not None:
            try:
                if timeout_seconds is None:
                    await asyncio.shield(running_task)
                else:
                    await asyncio.wait_for(
                        asyncio.shield(running_task), timeout=timeout_seconds
                    )
            except TimeoutError:
                return "Agent is still running.\n" + self._format_status(session, False)
            except asyncio.CancelledError:
                if session.status is not SessionStatus.INTERRUPTED:
                    raise

        return self._format_status(session)

    async def _run_subagent_once(
        self,
        subagent_name: str,
        task: str,
        on_subagent_created: Callable[[DelegatableAgent], Awaitable[None] | None]
        | None = None,
    ) -> str:
        if subagent_name not in self.agent.agent_config.subagents:
            raise ValueError(f"Agent type '{subagent_name}' is not configured.")

        agent_config = self.agent.config.agent.get(subagent_name)
        configured_type = agent_config.type if agent_config else "chat"

        if self.agent.session:
            child_session = self.agent.session.create_child_session(
                agent_name=subagent_name,
                agent_type=configured_type,
                workspace=str(self.agent.workspace) if self.agent.workspace else None,
            )
        else:
            child_session = AgentSession(
                agent_name=subagent_name,
                agent_type=configured_type,
                agent_source="static",
                workspace=str(self.agent.workspace) if self.agent.workspace else None,
                run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
            )

        candidate = child_session.create_agent(
            config_loader=self.agent.config_loader,
            io_adapter=self.agent.io_adapter,
        )
        try:
            subagent = self._as_delegatable(candidate, subagent_name)
        except BaseException:
            await self._discard_new_session(candidate.session)
            raise

        self._active_run_subagents[candidate.uuid] = subagent
        if self.agent.session:
            await self.agent.session.save()
        await self.agent.broadcast_agent_info()

        try:
            if on_subagent_created:
                result = on_subagent_created(subagent)
                if asyncio.iscoroutine(result):
                    await result

            async with subagent:
                if self.agent.session:
                    self.agent.session.record_subagent_call(subagent.name, task)
                result = await subagent.run_task(task)
                candidate.session.record_result(
                    result or "Task completed with no output."
                )
                return result or "Task completed with no output."
        finally:
            self._active_run_subagents.pop(candidate.uuid, None)
            candidate.session.close_session()
            await self.agent.broadcast_agent_info()

    @tool(sequential=True, enabled=_simple_tools_enabled)
    async def delegate_task(self, message: str, agent_type: str) -> str:
        """Delegate a task to a configured subagent and wait for its result."""
        return await self._run_subagent_once(agent_type, message)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def spawn_agent(self, task_name: str, message: str, agent_type: str) -> str:
        """Start a resumable task using a configured subagent.

        `task_name` must be a unique lowercase identifier and `message` must
        contain all context the subagent needs.
        """
        record = await self._spawn_record(task_name, message, agent_type)
        return "Agent spawned.\n" + self._format_status(record, False)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def followup_task(self, target: str, message: str) -> str:
        """Continue a completed, interrupted, or errored agent task.

        The existing agent session and message history are reused.
        """
        record = self._resolve_target(target)
        await self._start_followup(record, message)
        return "Follow-up started.\n" + self._format_status(record, False)

    @tool(enabled=_advanced_tools_enabled)
    async def wait_agent(self, target: str, timeout_seconds: float = 60) -> str:
        """Wait for an agent task and return its latest result.

        Timing out does not interrupt the task.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        return await self._wait_record(self._resolve_target(target), timeout_seconds)

    @tool(sequential=True, enabled=_advanced_tools_enabled)
    async def interrupt_agent(self, target: str) -> str:
        """Interrupt a running agent turn while preserving its session for follow-up."""
        record = self._resolve_target(target)
        running_task = record.running_task
        if running_task is None or running_task.done():
            return "Agent is not running.\n" + self._format_status(record, False)

        running_task.cancel()
        await asyncio.gather(running_task, return_exceptions=True)
        return "Agent interrupted.\n" + self._format_status(record, False)

    @tool(enabled=_advanced_tools_enabled)
    async def list_agents(self, status: SessionStatus | None = None) -> str:
        """List spawned agent tasks, optionally filtered by task status."""
        records = [
            record
            for record in self.tasks.values()
            if status is None or record.status == status
        ]
        if not records:
            return "No agent tasks."

        blocks = []
        for record in sorted(records, key=lambda item: item.created_at):
            block = self._format_status(record, False)
            if record.last_result:
                summary = record.last_result.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
