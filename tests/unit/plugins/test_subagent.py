import asyncio
import contextlib
from types import SimpleNamespace

import pytest

import arox.plugins.subagent as subagent_module
from arox.core.config import AgentConfig, Config
from arox.core.llm_base import DelegatableAgent
from arox.core.session import AgentSession, FileSessionStore
from arox.plugins.core import CorePlugin
from arox.plugins.subagent import (
    SubagentEvent,
    SubagentPlugin,
)


class _FakeDynamicAgent(DelegatableAgent):
    def __init__(
        self,
        parsed_config,
        io_adapter,
        session,
        workspace=None,
    ):
        super().__init__(
            parsed_config,
            io_adapter,
            session,
            workspace,
        )
        self.plugins = [CorePlugin(self)]

    async def __aenter__(self):
        # Minimal setup for testing subagent lifecycle
        self._tg = asyncio.TaskGroup()
        await self._stack.enter_async_context(self._tg)
        self._tg.create_task(self.io_adapter._process_io(self.adapter_io))
        await self._stack.enter_async_context(self.agent_io)
        await self._stack.enter_async_context(self.adapter_io)

        for plugin in self.plugins:
            await plugin.on_start()
            self._stack.push_async_callback(plugin.on_stop)
        return self

    async def run_task(self, task: str) -> str | None:
        return "task done"


class _MainAgent:
    def __init__(self, session: AgentSession, store: FileSessionStore):
        from arox.core.session import SessionManager

        self.name = "main"
        self.agent_session = session
        self.session_manager = SessionManager(store)
        self.session_manager.register_session_type(AgentSession)
        if session:
            session.manager = self.session_manager

        async def _fake_process_io(adapter_io):
            pass

        self.io_adapter = SimpleNamespace(_process_io=_fake_process_io)
        self.workspace = None
        self._stack = contextlib.AsyncExitStack()
        self._slots = {}
        self.parsed_config = Config(
            agent={
                "main": AgentConfig(type="chat", plugins=[], subagents=[]),
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
            return await result if hasattr(result, "__await__") else result
        return []

    async def save_session(self):
        pass

    async def load_child_agent_sessions(
        self, parent_session: AgentSession | None = None
    ) -> list[AgentSession]:
        parent = parent_session or self.agent_session
        if not isinstance(parent, AgentSession) or self.session_manager is None:
            return []
        child_owner_path = [*parent.owner_path, parent.id]
        children: list[AgentSession] = []
        for raw_child_ref in parent.children:
            child_ref: object = raw_child_ref
            child_id = (
                child_ref.id if isinstance(child_ref, AgentSession) else str(child_ref)
            )
            loaded = await self.session_manager.session_store.load_session(
                child_id, child_owner_path
            )
            if isinstance(loaded, AgentSession):
                children.append(loaded)
        return children

    async def unregister_child_session(
        self,
        child_session: AgentSession,
        parent_session: AgentSession | None = None,
    ) -> None:
        parent = parent_session or self.agent_session
        if not isinstance(parent, AgentSession) or self.session_manager is None:
            return
        parent.children = [cid for cid in parent.children if cid != child_session.id]
        await self.session_manager.session_store.delete_session(
            child_session.id, child_session.owner_path
        )


@pytest.mark.asyncio
async def test_dynamic_subagent_persists_spec_and_restores_on_reload(
    tmp_path, monkeypatch
):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    monkeypatch.setattr(
        subagent_module, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_session = AgentSession(id="main-session", agent_name="main")
    main_agent = _MainAgent(main_session, store)
    plugin = SubagentPlugin(main_agent)

    async with main_agent._stack:
        await plugin.create_subagent("planner", config={"description": "Plans work"})

    assert len(main_session.children) == 1
    child = await store.load_session(main_session.children[0], [main_session.id])
    assert isinstance(child, AgentSession)
    assert child.agent_name == "planner"
    assert child.agent_config.description == "Plans work"

    await store.save_session(main_session)
    loaded = await store.load_session("main-session")
    assert isinstance(loaded, AgentSession)

    reloaded_agent = _MainAgent(loaded, store)
    reloaded_plugin = SubagentPlugin(reloaded_agent)
    async with reloaded_agent._stack:
        await reloaded_plugin.on_start()
        assert list(reloaded_plugin.subagents) == ["planner"]
        restored = reloaded_plugin.subagents["planner"].session

    assert isinstance(restored, AgentSession)
    assert restored.id == loaded.children[0]
    assert reloaded_agent.parsed_config.agent["planner"].description == "Plans work"


@pytest.mark.asyncio
async def test_dynamic_subagent_can_reuse_existing_config(tmp_path, monkeypatch):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    monkeypatch.setattr(
        subagent_module, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_agent = _MainAgent(AgentSession(id="main-session", agent_name="main"), store)
    main_agent.parsed_config.agent["reviewer"] = AgentConfig(
        type="chat",
        description="Reviews code",
        model_params={"temperature": 0},
    )
    plugin = SubagentPlugin(main_agent)

    async with main_agent._stack:
        result = await plugin.create_subagent("reviewer")

    assert result == "Created subagent 'reviewer'."
    assert main_agent.session is not None
    child = await store.load_session(main_agent.session.children[0], ["main-session"])
    assert isinstance(child, AgentSession)
    assert child.agent_name == "reviewer"
    assert child.agent_config.description == "Reviews code"
    assert child.agent_config.model_params == {"temperature": 0}


@pytest.mark.asyncio
async def test_delete_dynamic_subagent_removes_session_and_does_not_restore(
    tmp_path, monkeypatch
):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    monkeypatch.setattr(
        subagent_module, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_session = AgentSession(id="main-session", agent_name="main")
    main_agent = _MainAgent(main_session, store)
    plugin = SubagentPlugin(main_agent)

    async with main_agent._stack:
        await plugin.create_subagent("planner")
        child_id = main_session.children[0]
        child = await store.load_session(child_id, [main_session.id])
        assert isinstance(child, AgentSession)
        assert await store.load_session(child.id, child.owner_path) is not None
        assert await plugin.delete_subagent("planner") == "Deleted subagent 'planner'."
        assert "planner" not in plugin.subagents
        assert main_session.children == []
        assert await store.load_session(child.id, child.owner_path) is None

    await store.save_session(main_session)
    loaded = await store.load_session("main-session")
    assert isinstance(loaded, AgentSession)
    reloaded_agent = _MainAgent(loaded, store)
    reloaded_plugin = SubagentPlugin(reloaded_agent)
    async with reloaded_agent._stack:
        await reloaded_plugin.on_start()
        assert reloaded_plugin.subagents == {}


@pytest.mark.asyncio
async def test_subagent_command_list_create_delete(tmp_path, monkeypatch):
    import arox.utils

    monkeypatch.setattr(
        arox.utils, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    monkeypatch.setattr(
        subagent_module, "import_class", lambda *_args, **_kwargs: _FakeDynamicAgent
    )
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_agent = _MainAgent(AgentSession(id="main-session", agent_name="main"), store)
    plugin = SubagentPlugin(main_agent)

    event = SubagentEvent.from_slash(
        "subagent", 'create planner chat {"description": "Plans work"}'
    )
    assert event.name == "planner"
    assert event.config == {"description": "Plans work"}

    async with main_agent._stack:
        assert (
            await plugin.handle_subagent_event(event) == "Created subagent 'planner'."
        )
        listing = await plugin.handle_subagent_event(SubagentEvent(action="list"))
        assert listing is not None
        assert "- planner (dynamic): Plans work" in listing
        assert (
            await plugin.handle_subagent_event(
                SubagentEvent(action="delete", name="planner")
            )
            == "Deleted subagent 'planner'."
        )
