import asyncio
import secrets
from enum import StrEnum
from typing import Any

from arox.core.agent_runtime import AgentRuntime
from arox.core.plugin import Plugin, tool
from arox.core.session import AgentSession
from arox.plugins.slots import SYSTEM_PROMPT


class SubagentMode(StrEnum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


class SubagentPlugin(Plugin):
    """Manage resumable child-agent tasks for a runtime."""

    def __init__(self, runtime):
        super().__init__(runtime)
        self.mode = SubagentMode.SIMPLE
        self.task_sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()
        self._stopping = False

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
        self._stopping = True
        session_manager = self.runtime.session.manager
        if session_manager is not None:
            await session_manager.stop_descendants(self.runtime.session)

    def _register_task_session(self, task_session: AgentSession) -> None:
        if task_session.target is None:
            raise ValueError("Subagent task session must have a target.")
        self.task_sessions[task_session.target] = task_session

    def _unregister_task_session(self, task_session: AgentSession) -> None:
        if task_session.target is not None:
            self.task_sessions.pop(task_session.target, None)

    def _resolve_session(self, target: str) -> AgentSession:
        try:
            return self.task_sessions[target]
        except KeyError:
            raise ValueError(f"Unknown agent target '{target}'.") from None

    def _create_target(self, task_name: str | None) -> str:
        target_name = task_name or "task"
        while True:
            short_id = secrets.token_urlsafe(6)
            target = f"/{self.runtime.name}/{target_name}-{short_id}"
            if target not in self.task_sessions:
                return target

    async def _spawn_task(
        self,
        task_name: str | None,
        message: str,
        subagent_name: str,
    ) -> AgentSession:
        async with self._lock:
            # create session
            if subagent_name not in self.runtime.agent_config.subagents:
                raise ValueError(f"Agent '{subagent_name}' is not configured.")

            target = self._create_target(task_name)
            task_session = await self.runtime.session.create_child_session(
                agent_name=subagent_name,
                agent_source="subagent",
                workspace=self.runtime.workspace,
                task_name=task_name,
                target=target,
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
        runtime = task_session.runtime
        try:
            if runtime is None:
                runtime = await task_session.ensure_runtime(
                    self.runtime.config_loader,
                    self.runtime.io_adapter,
                    AgentRuntime,
                )
            turn = runtime.start_message(message)
            if self.mode is SubagentMode.ADVANCED:
                self.runtime.background_tasks.register(task_session.target)
                turn.task.add_done_callback(
                    lambda _completed: self._notify_task_finished(task_session, runtime)
                )
            return turn.task
        except BaseException:
            if runtime is not None:
                await runtime.close()
            self._unregister_task_session(task_session)
            raise

    def _notify_task_finished(
        self, task_session: AgentSession, runtime: AgentRuntime
    ) -> None:
        if self._stopping:
            return
        turn = runtime.turn
        error = turn.error if turn is not None else None
        if error:
            status = f"failed ({task_session.format_error(error)})"
        else:
            status = "completed"
        self.runtime.background_tasks.complete(
            task_session.target,
            "Background subagent task finished.\n\n"
            f"Target: {task_session.target}\n"
            f"Description: {task_session.task_name or task_session.initial_message}\n"
            f"Status: {status}\n\n"
            f'Use wait_agent(target="{task_session.target}") to retrieve the '
            "result before continuing work that depends on it.",
        )

    def _format_task(
        self, task_session: AgentSession, include_result: bool = True
    ) -> str:
        lines = [
            f"- target: {task_session.target}",
            f"- task_name: {task_session.task_name}",
            f"- agent_name: {task_session.agent_name}",
        ]
        runtime = task_session.runtime
        turn = runtime.turn if runtime is not None else None
        if turn is not None:
            if turn.error:
                lines.append(f"- error: {task_session.format_error(turn.error)}")
            if include_result and turn.result:
                lines.extend(("", "Result:", turn.result.output))
        return "\n".join(lines)

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.SIMPLE,
    )
    async def delegate_task(self, message: str, agent_name: str) -> str:
        """Delegate a task to a configured subagent and wait for its result."""
        task_session = await self._spawn_task(None, message, agent_name)
        runtime = task_session.runtime
        try:
            turn = runtime.turn if runtime is not None else None
            if turn is None:
                return "Task completed with no output."
            await asyncio.gather(turn.task, return_exceptions=True)
            if turn.error:
                return f"Task failed: {task_session.format_error(turn.error)}"
            result = turn.result
            if result is None:
                return "Task completed with no output."
            return result.output
        finally:
            if runtime is not None:
                await runtime.close()

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def spawn_agent(
        self,
        task_name: str,
        message: str,
        agent_name: str,
    ) -> str:
        """Start a resumable task using a configured subagent.

        `task_name` must be a lowercase identifier used as a readable label and
        `message` must contain all context the subagent needs. Use the exact
        returned `target` with the task management tools.
        """
        task_session = await self._spawn_task(task_name, message, agent_name)
        return (
            "Agent spawned. You can call wait_agent at any time, and you will "
            "also be notified when this task finishes.\n"
            + self._format_task(task_session, False)
        )

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def followup_task(self, target: str, message: str) -> str:
        """Continue a completed, interrupted, or errored agent task.

        Use the exact target returned by spawn_agent or list_agents. The existing
        agent session and message history are reused.
        """
        async with self._lock:
            task_session = self._resolve_session(target)
            runtime = task_session.runtime
            if (
                runtime is not None
                and runtime.turn is not None
                and not runtime.turn.done
            ):
                raise ValueError(
                    f"Agent task '{task_session.target}' is already running."
                )

            await self._start_task(task_session, message)
        return (
            "Follow-up started. You can call wait_agent at any time, and you "
            "will also be notified when this task finishes.\n"
            + self._format_task(task_session, False)
        )

    @tool(enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED)
    async def wait_agent(self, target: str, timeout_seconds: float = 600) -> str:
        """Wait for an agent task and return its latest result.

        Use the exact target returned by spawn_agent or list_agents. Timing out
        does not interrupt the task.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        task_session = self._resolve_session(target)
        runtime = task_session.runtime
        turn = runtime.turn if runtime is not None else None

        if turn is None:
            return self._format_task(task_session)
        done, _ = await asyncio.wait({turn.task}, timeout=timeout_seconds)
        if not done:
            return "Agent is still running.\n" + self._format_task(task_session, False)

        await asyncio.gather(turn.task, return_exceptions=True)
        self.runtime.background_tasks.observe(target)
        outcome = self._format_task(task_session)
        await runtime.close()
        return outcome

    @tool(
        sequential=True,
        enabled=lambda plugin: plugin.mode is SubagentMode.ADVANCED,
    )
    async def interrupt_agent(self, target: str) -> str:
        """Interrupt the targeted turn while preserving its agent session."""
        task_session = self._resolve_session(target)
        async with self._lock:
            runtime = task_session.runtime
            turn = runtime.turn if runtime is not None else None
            if turn is None or turn.done:
                return "Agent is not running.\n" + self._format_task(
                    task_session, False
                )
            await turn.cancel()
            self.runtime.background_tasks.observe(target)
            outcome = self._format_task(task_session)
            await runtime.close()
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
            runtime = task_session.runtime
            turn = runtime.turn if runtime is not None else None
            if turn is not None and turn.result and isinstance(turn.result.output, str):
                summary = turn.result.output.replace("\n", " ")[:200]
                block += f"\n- result_summary: {summary}"
            blocks.append(block)
        return "\n\n".join(blocks)
