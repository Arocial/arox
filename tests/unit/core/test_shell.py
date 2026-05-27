from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.session import AgentSession, AppSession
from arox.plugins.session import ForkEvent, SessionPlugin


def _make_plugin(main_agent_session: AgentSession):
    session = AppSession(
        id="s1",
        main_agent="test",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        agent_sessions={main_agent_session.agent_name: main_agent_session},
    )

    class MockAgent:
        def __init__(self):
            self.name = main_agent_session.agent_name
            self.agent_session = main_agent_session
            self.parsed_config = SimpleNamespace(
                app=SimpleNamespace(session_max_age_days=30)
            )

    agent = MockAgent()
    plugin = SessionPlugin(agent)
    plugin.session = session

    async def fork_session(agent_name: str, event_index: int) -> str:
        return f"forked:{agent_name}:{event_index}"

    plugin.fork_session = fork_session  # ty: ignore[invalid-assignment]
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
    assert "forked:main:2" in msg

    msg = await plugin.handle_fork(ForkEvent(n=2))
    assert "forked:main:0" in msg


@pytest.mark.asyncio
async def test_handle_fork_absolute_anchor_check():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    plugin = _make_plugin(ag)

    # @0 is a user-turn anchor → succeeds.
    msg = await plugin.handle_fork(ForkEvent(n=None, event_index=0))
    assert "forked:main:0" in msg

    # @1 is an agent_step, not an anchor → rejected.
    msg = await plugin.handle_fork(ForkEvent(n=None, event_index=1))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_fork_not_enough_history():
    ag = AgentSession(agent_name="main")
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(n=1))
    assert "not enough history" in msg
