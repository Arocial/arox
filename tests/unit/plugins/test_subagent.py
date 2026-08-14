import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arox.core.agent_runtime import AgentRuntime
from arox.core.config import AgentConfig, Config
from arox.core.runner import TaskRunner
from arox.core.session import (
    AgentSession,
    FileSessionStore,
)
from arox.plugins.core import CorePlugin
from arox.plugins.slots import SYSTEM_PROMPT
from arox.plugins.subagent import (
    SubagentMode,
    SubagentPlugin,
)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")


class _FakeDynamicAgent(AgentRuntime):
    def __init__(
        self,
        parent_config_loader,
        io_adapter,
        session,
    ):
        super().__init__(
            parent_config_loader,
            io_adapter,
            session,
        )
        self.plugins = [CorePlugin(self)]
        self.received_tasks: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run_turn(self, user_input=None):
        task = str(user_input or "")
        self.received_tasks.append(task)
        self.started.set()
        if task.startswith("fail"):
            raise RuntimeError("task failed")
        if task.startswith("block"):
            await self.release.wait()
        output = f"task done: {task}"
        return SimpleNamespace(output=output)


class _HostAgent:
    async def broadcast_session_tree(self):
        pass

    def __init__(
        self,
        session: AgentSession,
        store: FileSessionStore,
    ):
        from arox.core.session import SessionManager

        self.uuid = "test-uuid"
        self.name = "main"
        self.agent_session = session
        self.session_manager = SessionManager(store)
        self.session_manager.register_session_type(AgentSession)
        session.manager = self.session_manager

        def register_host(host):
            self.io_adapter.hosts[host.uuid] = host

        def unregister_host(host):
            self.io_adapter.hosts.pop(host.uuid, None)

        self.io_adapter = SimpleNamespace(
            handle_event=AsyncMock(),
            on_endpoint_closed=AsyncMock(),
            register_host=register_host,
            unregister_host=unregister_host,
            hosts={},
        )
        self.agent_ep = SimpleNamespace(send=AsyncMock())
        self.workspace = None
        self._stack = contextlib.AsyncExitStack()
        self._slots = {}
        self.config = Config(
            agent={
                "main": AgentConfig(
                    plugins=[],
                    subagents=["planner", "reviewer"],
                ),
                "planner": AgentConfig(description="Plans work"),
                "reviewer": AgentConfig(description="Reviews code"),
            }
        )

        def make_config_loader(workspace=None):
            return SimpleNamespace(
                current_config=self.config,
                for_workspace=make_config_loader,
                workspace=workspace,
            )

        self.config_loader = make_config_loader()
        self.agent_config = self.config.agent["main"]

    @property
    def session(self) -> AgentSession:
        return self.agent_session

    def provide_slot(self, slot, provider):
        self._slots.setdefault(slot, []).append(provider)

    async def invoke_slot(self, slot, *args, **kwargs):
        providers = self._slots.get(slot, [])
        if slot.aggregator.name == "FIRST":
            if not providers:
                return None
            result = providers[0](*args, **kwargs)
            return await result if asyncio.iscoroutine(result) else result
        return []


@pytest.fixture
def agent_factory(tmp_path, monkeypatch):
    monkeypatch.setattr("arox.core.runner.AgentRuntime", _FakeDynamicAgent)
    store = FileSessionStore()
    store.base_dir = tmp_path / "sessions"

    def create():
        session = AgentSession(path=["main-session"], agent_name="main")
        return _HostAgent(session, store)

    return create, store


def _advanced_plugin(agent):
    plugin = SubagentPlugin(agent)
    plugin.configure({"mode": "advanced"})
    return plugin


@pytest.mark.asyncio
async def test_simple_mode_exposes_only_delegate_and_waits_for_result(agent_factory):
    create_agent, store = agent_factory
    main_agent = create_agent()
    plugin = SubagentPlugin(main_agent)

    try:
        assert plugin.mode is SubagentMode.SIMPLE
        assert plugin.config == {}

        toolset = plugin._build_toolset()
        assert toolset is not None
        assert set(toolset.tools) == {"delegate_task"}

        prompt = main_agent._slots[SYSTEM_PROMPT][0]()
        assert "delegate tasks synchronously" in prompt
        assert "delegate_task" in prompt
        assert "returns the completed result" in prompt
        assert "spawn_agent" not in prompt
        assert "wait_agent" not in prompt

        response = await plugin.delegate_task("make a plan", "planner")

        assert response == "task done: make a plan"
        assert len(plugin.task_sessions) == 1
        task_session = next(iter(plugin.task_sessions.values()))
        assert task_session.runner is None
        assert len(main_agent.session.children) == 1
        call = main_agent.session.events[-1]
        assert call.event_type == "subagent_call"
        assert call.subagent == "planner"
        assert call.task == "make a plan"

        child_session = await store.load_session(
            [main_agent.session.id, main_agent.session.children[0]]
        )
        assert child_session is not None
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_simple_mode_uses_registered_task_lifecycle(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = SubagentPlugin(main_agent)
    delegation = asyncio.create_task(
        plugin.delegate_task("block until released", "planner")
    )

    try:
        while not plugin.task_sessions:
            await asyncio.sleep(0.01)
        task_session = next(iter(plugin.task_sessions.values()))
        subagent = task_session.runner.runtime
        assert isinstance(subagent, _FakeDynamicAgent)
        await subagent.started.wait()

        assert await plugin.delegate_task("second task", "reviewer") == (
            "task done: second task"
        )

        subagent.release.set()
        assert await delegation == "task done: block until released"
        assert len(plugin.task_sessions) == 2
        assert all(session.runner is None for session in plugin.task_sessions.values())
    finally:
        for task_session in plugin.task_sessions.values():
            runner = task_session.runner
            if runner is not None and isinstance(runner.runtime, _FakeDynamicAgent):
                runner.runtime.release.set()
        await asyncio.gather(delegation, return_exceptions=True)
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_advanced_mode_exposes_task_management_tools(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        assert plugin.mode is SubagentMode.ADVANCED
        assert plugin.config["mode"] == "advanced"

        toolset = plugin._build_toolset()
        assert toolset is not None
        assert set(toolset.tools) == {
            "followup_task",
            "interrupt_agent",
            "list_agents",
            "spawn_agent",
            "wait_agent",
        }

        prompt = main_agent._slots[SYSTEM_PROMPT][0]()
        assert "resumable tasks" in prompt
        assert "spawn_agent" in prompt
        assert "delegate_task" not in prompt
        assert "wait_agent" in prompt
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_spawn_and_wait_releases_runner_and_preserves_session(agent_factory):
    create_agent, store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        response = await plugin.spawn_agent("make_plan", "make a plan", "planner")
        assert "Agent spawned." in response

        task_session = plugin._resolve_session("make_plan")
        result = await plugin.wait_agent(task_session.task_id)

        assert "task done: make a plan" in result
        assert task_session.runtime is None
        assert task_session.runner is None
        assert len(main_agent.session.children) == 1

        child_session = await store.load_session(
            [main_agent.session.id, main_agent.session.children[0]]
        )
        assert child_session is not None
        assert child_session.task_name == "make_plan"
        persisted = child_session.model_dump()
        assert "last_message" not in persisted
        assert "result" not in persisted
        assert "error" not in persisted
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_spawn_starts_runtime_and_retains_session(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)
    try:
        task_session = await plugin._spawn_task(
            "make_plan",
            "make a plan",
            "planner",
        )
        assert isinstance(task_session.runner.runtime, AgentRuntime)
        await plugin.wait_agent(task_session.task_id)

        assert task_session.runner is None
        assert plugin._resolve_session("make_plan") is task_session
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_spawn_failure_unregisters_task_and_preserves_child_session(
    agent_factory, monkeypatch
):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    async def fail_setup(_runner: TaskRunner) -> AgentRuntime:
        raise RuntimeError("setup failed")

    try:
        monkeypatch.setattr(TaskRunner, "start_runtime", fail_setup)
        with pytest.raises(RuntimeError, match="setup failed"):
            await plugin._spawn_task("make_plan", "make a plan", "planner")

        assert plugin.task_sessions == {}
        assert len(main_agent.session.children) == 1
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_task(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        await plugin.spawn_agent("slow_review", "block until released", "reviewer")
        task_session = plugin._resolve_session("slow_review")
        assert task_session.runner.task is not None

        # Wait for agent to start
        while task_session.runner is None:
            await asyncio.sleep(0.01)
        subagent = task_session.runner.runtime
        assert isinstance(subagent, _FakeDynamicAgent)
        await subagent.started.wait()

        result = await plugin.wait_agent("slow_review", timeout_seconds=0.01)

        assert "Agent is still running." in result
        assert task_session.runner.task is not None
        assert not task_session.runner.task.cancelled()

        subagent.release.set()
        completed = await plugin.wait_agent("slow_review")
        assert "Result:" in completed
        assert task_session.runner is None
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_wait_reports_runner_exception(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        await plugin.spawn_agent("failed_review", "fail review", "reviewer")

        result = await plugin.wait_agent("failed_review")

        task_session = plugin._resolve_session("failed_review")
        assert task_session.runner is None
        assert "error: RuntimeError: task failed" in result
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_interrupt_then_followup_reuses_session_with_new_runner(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        await plugin.spawn_agent("review_patch", "block first review", "reviewer")
        task_session = plugin._resolve_session("review_patch")
        original_session_id = task_session.id

        while task_session.runner is None:
            await asyncio.sleep(0.01)
        subagent = task_session.runner.runtime
        assert isinstance(subagent, _FakeDynamicAgent)
        await subagent.started.wait()

        interrupted = await plugin.interrupt_agent("review_patch")
        assert "Task interrupted." in interrupted
        assert task_session.runner is None

        followup = await plugin.followup_task(
            "/main/review_patch", "review the updated tests"
        )
        assert "Follow-up started." in followup
        assert isinstance(task_session.runner, TaskRunner)
        followup_subagent = task_session.runner.runtime
        assert isinstance(followup_subagent, _FakeDynamicAgent)
        assert followup_subagent is not subagent
        assert task_session.runner.result is None
        assert task_session.runner.error is None
        completed = await plugin.wait_agent(task_session.task_id)

        assert "review the updated tests" in completed
        assert task_session.runner is None
        assert task_session.id == original_session_id
        assert subagent.received_tasks == [
            "block first review",
        ]
        assert followup_subagent.received_tasks == ["review the updated tests"]
        assert len(main_agent.session.children) == 1
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_list_agents_includes_idle_and_running_sessions(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        await plugin.spawn_agent("make_plan", "make a plan", "planner")
        await plugin.wait_agent("make_plan")
        await plugin.spawn_agent("slow_review", "block review", "reviewer")
        slow_task_session = plugin._resolve_session("slow_review")
        while slow_task_session.runner is None:
            await asyncio.sleep(0.01)
        assert isinstance(slow_task_session.runner.runtime, _FakeDynamicAgent)
        await slow_task_session.runner.runtime.started.wait()

        agents = await plugin.list_agents()

        assert "/main/make_plan" in agents
        assert "/main/slow_review" in agents
        assert plugin._resolve_session("make_plan").runner is None

        slow_task_session.runner.runtime.release.set()
        await plugin.wait_agent("slow_review")
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_spawn_validates_name_type_and_duplicate_task(agent_factory):
    create_agent, _store = agent_factory
    main_agent = create_agent()
    plugin = _advanced_plugin(main_agent)

    try:
        with pytest.raises(ValueError, match="task_name"):
            await plugin.spawn_agent("Bad-Name", "task", "planner")
        with pytest.raises(ValueError, match="not configured"):
            await plugin.spawn_agent("unknown_type", "task", "unknown")

        await plugin.spawn_agent("first_task", "block first", "planner")
        first_task_session = plugin._resolve_session("first_task")
        while first_task_session.runner is None:
            await asyncio.sleep(0.01)
        assert isinstance(first_task_session.runner.runtime, _FakeDynamicAgent)
        await first_task_session.runner.runtime.started.wait()

        await plugin.spawn_agent("second_task", "task", "reviewer")
        second_result = await plugin.wait_agent("second_task")
        assert "task done: task" in second_result
        with pytest.raises(ValueError, match="already exists"):
            await plugin.spawn_agent("first_task", "task", "planner")

        first_task_session.runner.runtime.release.set()
        await plugin.wait_agent("first_task")
    finally:
        await plugin.on_stop()


@pytest.mark.asyncio
async def test_on_start_restores_task_without_process_state(agent_factory):
    create_agent, store = agent_factory
    first_main = create_agent()
    first_plugin = _advanced_plugin(first_main)

    await first_plugin.spawn_agent("slow_plan", "block planning", "planner")
    first_task_session = first_plugin._resolve_session("slow_plan")
    while first_task_session.runner is None:
        await asyncio.sleep(0.01)
    assert isinstance(first_task_session.runner.runtime, _FakeDynamicAgent)
    await first_task_session.runner.runtime.started.wait()

    restored_session = await store.load_session(first_main.session.path)
    assert isinstance(restored_session, AgentSession)
    restored_main = _HostAgent(restored_session, store)
    restored_plugin = _advanced_plugin(restored_main)
    try:
        await restored_plugin.on_start()
        restored = restored_plugin._resolve_session("slow_plan")

        assert restored.runner is None
    finally:
        first_task_session.runner.runtime.release.set()
        await restored_plugin.on_stop()
        await first_plugin.on_stop()
