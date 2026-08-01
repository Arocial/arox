import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from arox.core.llm_base import DelegatableAgent, create_agent
from arox.core.plugin import Plugin, tool
from arox.core.session import AgentSession
from arox.plugins.slots import RUN_SUBAGENT, SUBAGENTS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TASK_METADATA_KEY = "subagent_task"
_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SubagentMode(StrEnum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


class SubagentTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERRORED = "errored"


@dataclass(kw_only=True)
class SubagentTask:
    task_id: str
    task_name: str
    target: str
    agent_type: str
    initial_message: str
    last_message: str
    session: AgentSession
    status: SubagentTaskStatus = SubagentTaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent: DelegatableAgent | None = None
    stack: AsyncExitStack | None = None
    running_task: asyncio.Task[str | None] | None = None


class SubagentPlugin(Plugin):
    """Manage resumable child-agent tasks for an agent."""

    def __init__(self, agent):
        super().__init__(agent)
        self.mode = SubagentMode.SIMPLE
        self.subagents: dict[str, DelegatableAgent] = {}
        self.tasks: dict[str, SubagentTask] = {}
        self._tasks_by_name: dict[str, str] = {}
        self._tasks_by_target: dict[str, str] = {}
        self._lock = asyncio.Lock()

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

            metadata = child_session.extra.get(_TASK_METADATA_KEY)
            if not isinstance(metadata, dict):
                continue

            try:
                record = self._record_from_metadata(child_session, metadata)
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid subagent task metadata for session %s",
                    child_session.id,
                    exc_info=True,
                )
                continue

            if record.status in (
                SubagentTaskStatus.PENDING,
                SubagentTaskStatus.RUNNING,
            ):
                record.status = SubagentTaskStatus.INTERRUPTED
                record.error = "Parent process stopped before the task completed."
                child_session.status = "closed"
                await self._persist(record)

            if record.task_id in self.tasks or record.task_name in self._tasks_by_name:
                logger.warning(
                    "Ignoring duplicate restored subagent task %s", record.target
                )
                continue
            self._register_record(record)

    async def on_stop(self) -> None:
        running = [
            record.running_task
            for record in self.tasks.values()
            if record.running_task and not record.running_task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

        for record in list(self.tasks.values()):
            await self._close_agent(record)

    def _record_from_metadata(
        self, session: AgentSession, metadata: dict[str, Any]
    ) -> SubagentTask:
        status = SubagentTaskStatus(metadata["status"])

        return SubagentTask(
            task_id=metadata["task_id"],
            task_name=metadata["task_name"],
            target=metadata["target"],
            agent_type=metadata["agent_type"],
            initial_message=metadata["initial_message"],
            last_message=metadata["last_message"],
            session=session,
            status=status,
            result=metadata.get("result"),
            error=metadata.get("error"),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    def _register_record(self, record: SubagentTask) -> None:
        self.tasks[record.task_id] = record
        self._tasks_by_name[record.task_name] = record.task_id
        self._tasks_by_target[record.target] = record.task_id

    def _task_metadata(self, record: SubagentTask) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "task_name": record.task_name,
            "target": record.target,
            "agent_type": record.agent_type,
            "status": record.status,
            "initial_message": record.initial_message,
            "last_message": record.last_message,
            "result": record.result,
            "error": record.error,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    async def _persist(self, record: SubagentTask) -> None:
        record.updated_at = datetime.now(UTC)
        record.session.extra[_TASK_METADATA_KEY] = self._task_metadata(record)
        await record.session.save()

    @staticmethod
    def _as_delegatable(candidate: Any, agent_type: str) -> DelegatableAgent:
        if not isinstance(candidate, DelegatableAgent):
            raise TypeError(
                f"Agent type '{agent_type}' does not support delegated tasks."
            )
        return candidate

    async def _discard_new_session(self, session: AgentSession) -> None:
        if session.id in self.agent.session.children:
            self.agent.session.children.remove(session.id)
            await self.agent.session.save()

    async def _activate_agent(
        self,
        record: SubagentTask,
        subagent: DelegatableAgent | None = None,
    ) -> DelegatableAgent:
        if record.agent is not None:
            return record.agent

        if subagent is None:
            candidate = create_agent(
                name=record.agent_type,
                config_loader=self.agent.config_loader,
                io_adapter=self.agent.io_adapter,
                session=record.session,
            )
            subagent = self._as_delegatable(candidate, record.agent_type)

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            await stack.enter_async_context(subagent)
        except BaseException:
            await stack.aclose()
            raise

        record.agent = subagent
        record.stack = stack
        self.subagents[subagent.uuid] = subagent
        await self.agent.broadcast_agent_info()
        return subagent

    async def _close_agent(self, record: SubagentTask) -> None:
        subagent = record.agent
        stack = record.stack
        record.agent = None
        record.stack = None

        if stack is not None:
            await stack.aclose()
        if subagent is not None:
            self.subagents.pop(subagent.uuid, None)
            if hasattr(self.agent.io_adapter, "hosts"):
                self.agent.io_adapter.hosts.pop(subagent.uuid, None)

    def _resolve_target(self, target: str) -> SubagentTask:
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
            record.status in (SubagentTaskStatus.PENDING, SubagentTaskStatus.RUNNING)
            for record in self.tasks.values()
        )

    async def _spawn_record(
        self,
        task_name: str,
        message: str,
        agent_type: str,
        on_subagent_created: Callable[[DelegatableAgent], Awaitable[None] | None]
        | None = None,
    ) -> SubagentTask:
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

            candidate = create_agent(
                name=agent_type,
                config_loader=self.agent.config_loader,
                io_adapter=self.agent.io_adapter,
                parent_session=self.agent.session,
                agent_source="static",
                workspace=self.agent.workspace,
            )
            record = SubagentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                task_name=task_name,
                target=f"/{self.agent.name}/{task_name}",
                agent_type=agent_type,
                initial_message=message,
                last_message=message,
                session=candidate.session,
            )
            self._register_record(record)

            try:
                subagent = self._as_delegatable(candidate, agent_type)
                await self._activate_agent(record, subagent)
                if on_subagent_created:
                    result = on_subagent_created(subagent)
                    if asyncio.iscoroutine(result):
                        await result
                await self._persist(record)
                await self.agent.session.save()
            except BaseException:
                self.tasks.pop(record.task_id, None)
                self._tasks_by_name.pop(record.task_name, None)
                self._tasks_by_target.pop(record.target, None)
                await self._close_agent(record)
                await self._discard_new_session(record.session)
                raise

            record.running_task = asyncio.create_task(
                self._run_turn(record, message),
                name=f"subagent:{record.target}",
            )
            return record

    async def _run_turn(self, record: SubagentTask, message: str) -> str | None:
        subagent = await self._activate_agent(record)
        record.status = SubagentTaskStatus.RUNNING
        record.result = None
        record.error = None
        record.last_message = message
        record.session.status = "active"
        self.agent.session.record_subagent_call(record.agent_type, message)
        await self._persist(record)
        await self.agent.broadcast_agent_info()

        try:
            result = await subagent.run_task(message)
        except asyncio.CancelledError:
            record.status = SubagentTaskStatus.INTERRUPTED
            record.error = "Task interrupted."
            raise
        except Exception as exc:
            logger.exception("Subagent task %s failed", record.target)
            record.status = SubagentTaskStatus.ERRORED
            record.error = f"{type(exc).__name__}: {exc}"
        else:
            record.status = SubagentTaskStatus.COMPLETED
            record.result = result or "Task completed with no output."
        finally:
            record.session.status = "closed"
            record.running_task = None
            await self._persist(record)
            await self.agent.broadcast_agent_info()

        return record.result

    async def _start_followup(self, record: SubagentTask, message: str) -> None:
        async with self._lock:
            if record.running_task and not record.running_task.done():
                raise ValueError(f"Agent task '{record.target}' is already running.")
            if self._running_count() >= self.agent.agent_config.max_parallel_subagents:
                raise ValueError(
                    "Maximum parallel subagents reached: "
                    f"{self.agent.agent_config.max_parallel_subagents}."
                )

            await self._activate_agent(record)
            record.status = SubagentTaskStatus.PENDING
            record.result = None
            record.error = None
            record.last_message = message
            record.session.status = "active"
            await self._persist(record)
            record.running_task = asyncio.create_task(
                self._run_turn(record, message),
                name=f"subagent:{record.target}:followup",
            )

    def _format_status(self, record: SubagentTask, include_result: bool = True) -> str:
        lines = [
            f"- task_id: {record.task_id}",
            f"- target: {record.target}",
            f"- agent_type: {record.agent_type}",
            f"- status: {record.status}",
        ]
        if record.error:
            lines.append(f"- error: {record.error}")
        if include_result and record.result:
            lines.extend(("", "Result:", record.result))
        return "\n".join(lines)

    async def _wait_record(
        self, record: SubagentTask, timeout_seconds: float | None
    ) -> str:
        running_task = record.running_task
        if running_task is not None:
            try:
                if timeout_seconds is None:
                    await asyncio.shield(running_task)
                else:
                    await asyncio.wait_for(
                        asyncio.shield(running_task), timeout=timeout_seconds
                    )
            except TimeoutError:
                return "Agent is still running.\n" + self._format_status(record, False)
            except asyncio.CancelledError:
                if record.status is not SubagentTaskStatus.INTERRUPTED:
                    raise

        return self._format_status(record)

    async def _run_subagent_once(
        self,
        subagent_name: str,
        task: str,
        on_subagent_created: Callable[[DelegatableAgent], Awaitable[None] | None]
        | None = None,
    ) -> str:
        if subagent_name not in self.agent.agent_config.subagents:
            raise ValueError(f"Agent type '{subagent_name}' is not configured.")

        candidate = create_agent(
            name=subagent_name,
            config_loader=self.agent.config_loader,
            io_adapter=self.agent.io_adapter,
            parent_session=self.agent.session,
            agent_source="static",
            workspace=self.agent.workspace,
        )
        try:
            subagent = self._as_delegatable(candidate, subagent_name)
        except BaseException:
            await self._discard_new_session(candidate.session)
            raise

        self.subagents[subagent.uuid] = subagent
        await self.agent.session.save()
        await self.agent.broadcast_agent_info()

        try:
            if on_subagent_created:
                result = on_subagent_created(subagent)
                if asyncio.iscoroutine(result):
                    await result

            async with subagent:
                self.agent.session.record_subagent_call(subagent.name, task)
                result = await subagent.run_task(task)
                return result or "Task completed with no output."
        finally:
            subagent.close_session()
            self.subagents.pop(subagent.uuid, None)
            if hasattr(self.agent.io_adapter, "hosts"):
                self.agent.io_adapter.hosts.pop(subagent.uuid, None)
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
    async def list_agents(self, status: SubagentTaskStatus | None = None) -> str:
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
            if record.result:
                summary = record.result.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
