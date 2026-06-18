import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arox.core.config import AgentConfig, Config
from arox.core.llm_base import DelegatableAgent
from arox.core.session import AgentSession, FileSessionStore
from arox.plugins.core import CorePlugin
from arox.plugins.subagent import (
    SubagentEvent,
    SubagentPlugin,
)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")


class _FakeDynamicAgent(DelegatableAgent):
    def __init__(
        self,
        parsed_config,
        io_adapter,
        session,
    ):
        super().__init__(
            parsed_config,
            io_adapter,
            session,
        )
        self.plugins = [CorePlugin(self)]

    async def __aenter__(self):
        # Minimal setup for testing subagent lifecycle
        self._tg = asyncio.TaskGroup()
        await self._stack.enter_async_context(self._tg)
        if hasattr(self.io_adapter, "_process_io"):
            self._tg.create_task(self.io_adapter._process_io(self.adapter_io))
        await self._stack.enter_async_context(self.agent_io)
        await self._stack.enter_async_context(self.adapter_io)

        for plugin in self.plugins:
            await plugin.on_start()
            self._stack.push_async_callback(plugin.on_stop)
        return self

    async def run_task(self, task: str) -> str:
        return f"task done: {task}"


class _MainAgent:
    def __init__(self, session: AgentSession, store: FileSessionStore):
        from arox.core.session import SessionManager

        self.uuid = "test-uuid"
        self.name = "main"
        self.agent_session = session
        self.session_manager = SessionManager(store)
        self.session_manager.register_session_type(AgentSession)
        if session:
            session.manager = self.session_manager

        async def _fake_process_io(adapter_io):
            pass

        self.io_adapter = SimpleNamespace(_process_io=_fake_process_io, hosts={})
        self.agent_io = SimpleNamespace(send=AsyncMock())
        self.workspace = None
        self._stack = contextlib.AsyncExitStack()
        self._slots = {}
        self.parsed_config = Config(
            agent={
                "main": AgentConfig(
                    type="chat", plugins=[], subagents=["planner", "reviewer"]
                ),
                "planner": AgentConfig(type="chat", description="Plans work"),
                "reviewer": AgentConfig(type="chat", description="Reviews code"),
            }
        )
        self.agent_config = self.parsed_config.agent["main"]

    @property
    def session(self) -> AgentSession | None:
        return self.agent_session

    @session.setter
    def session(self, value: AgentSession | None) -> None:
        self.agent_session = value

    def provide_slot(self, slot, provider):
        self._slots.setdefault(slot, []).append(provider)

    async def invoke_slot(self, slot, *args, **kwargs):
        providers = self._slots.get(slot, [])
        if slot.aggregator.name == "FIRST":
            if not providers:
                return None
            result = providers[0](*args, **kwargs)
            import asyncio

            return await result if asyncio.iscoroutine(result) else result
        return []

    async def save_session(self):
        pass


@pytest.mark.asyncio
async def test_delegate_to_subagent_creates_and_destroys(tmp_path, monkeypatch):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_session = AgentSession(path=["main-session"], agent_name="main")
    main_agent = _MainAgent(main_session, store)
    plugin = SubagentPlugin(main_agent)

    async with main_agent._stack:
        result = await plugin.delegate_to_subagent("planner", "make a plan")

    assert result == "task done: make a plan"
    assert len(main_session.children) == 1
    child_session = await store.load_session(
        [main_session.id, main_session.children[0]]
    )
    assert child_session is not None
    assert child_session.status == "closed"

    assert not plugin.active_subagents


@pytest.mark.asyncio
async def test_dispatch_background_task_creates_and_destroys(tmp_path, monkeypatch):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_session = AgentSession(path=["main-session"], agent_name="main")
    main_agent = _MainAgent(main_session, store)
    plugin = SubagentPlugin(main_agent)

    async with main_agent._stack:
        res_msg = await plugin.dispatch_background_task("reviewer", "review this")
        assert "Task dispatched to reviewer" in res_msg

        # Parse task ID
        import re

        match = re.search(r"Task ID: (task_[0-9a-f]+)", res_msg)
        assert match
        task_id = match.group(1)

        # wait for task to finish
        task = plugin.background_tasks[task_id]
        await task

        status = await plugin.check_task_status(task_id)
        assert "Task Completed. Result:" in status
        assert "review this" in status

        assert len(main_session.children) == 1
        child_session = await store.load_session(
            [main_session.id, main_session.children[0]]
        )
        assert child_session is not None
        assert child_session.status == "closed"

        assert not plugin.active_subagents


@pytest.mark.asyncio
async def test_subagent_command_list_call(tmp_path, monkeypatch):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_agent = _MainAgent(
        AgentSession(path=["main-session"], agent_name="main"), store
    )
    plugin = SubagentPlugin(main_agent)

    event_call = SubagentEvent.from_slash("subagent", "call planner make a plan")
    assert event_call.name == "planner"
    assert event_call.task == "make a plan"

    async with main_agent._stack:
        listing = await plugin.handle_subagent_event(SubagentEvent(action="list"))
        assert listing is not None
        assert "- planner: Plans work" in listing
        assert "- reviewer: Reviews code" in listing

        result = await plugin.handle_subagent_event(event_call)
        assert result == "task done: make a plan"
