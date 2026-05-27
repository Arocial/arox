from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.session import AppSession
from arox.plugins.session import AgentSession, ForkEvent, SessionPlugin


def _make_plugin(main_agent_session: AgentSession):
    session = AppSession(
        id="s1",
        main_agent="test",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    class MockAgent:
        def __init__(self):
            self.name = main_agent_session.agent_name
            self.parsed_config = SimpleNamespace(
                app=SimpleNamespace(session_max_age_days=30)
            )
            self.plugins = []

        async def invoke_slot(self, slot, *args, **kwargs):
            return []

    agent = MockAgent()
    plugin = SessionPlugin(agent)
    plugin.agent_session = main_agent_session
    plugin.app_session = session
    agent.plugins.append(plugin)

    # Mock session_store.save_session
    async def mock_save_session(s):
        pass

    plugin.session_store.save_session = mock_save_session  # ty: ignore[invalid-assignment]

    return plugin


def test_fork_event_parsing():
    assert ForkEvent.from_slash("fork", "").n == 1
    assert ForkEvent.from_slash("fork", "3").n == 3
    e = ForkEvent.from_slash("fork", "@5")
    assert e.n is None and e.event_index == 5
    # Bad input falls back to default.
    assert ForkEvent.from_slash("fork", "abc").n == 1


@pytest.mark.asyncio
async def test_handle_fork_relative_success():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    ag.add_event("user_input", {"text": "again"})
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(n=1))
    assert "Forked at event @2" in msg

    msg = await plugin.handle_fork(ForkEvent(n=2))
    assert "Forked at event @0" in msg


@pytest.mark.asyncio
async def test_handle_fork_absolute_anchor_check():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    plugin = _make_plugin(ag)

    # @0 is a user-turn anchor → succeeds.
    msg = await plugin.handle_fork(ForkEvent(n=None, event_index=0))
    assert "Forked at event @0" in msg

    # @1 is an agent_step, not an anchor → rejected.
    msg = await plugin.handle_fork(ForkEvent(n=None, event_index=1))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_fork_not_enough_history():
    ag = AgentSession(agent_name="main")
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(n=1))
    assert "not enough history" in msg
