from types import SimpleNamespace

import pytest

from arox.core.completion import CompletionRequest
from arox.core.config import AgentConfig
from arox.core.session import AgentSession, StepEvent, UserInputEvent
from arox.plugins.core import CorePlugin, ForkEvent


def _make_plugin(main_agent_session: AgentSession):
    class MockAgent:
        session: AgentSession | None = None
        owner: AgentSession | None = None

        def __init__(self):
            self.name = main_agent_session.agent_name
            self.agent_config = AgentConfig(type="chat")
            self.parsed_config = SimpleNamespace(
                app=SimpleNamespace(session_max_age_days=30)
            )
            self.plugins = []
            self.session_manager = None

        async def invoke_slot(self, slot, *args, **kwargs):
            return []

    agent = MockAgent()
    plugin = CorePlugin(agent)
    agent.session = main_agent_session
    agent.owner = AgentSession(agent_name="parent")
    agent.plugins.append(plugin)

    # Stub it with a no-op save.
    async def mock_save_session(s):
        pass

    agent.session_manager = SimpleNamespace(
        session_store=SimpleNamespace(save_session=mock_save_session)
    )

    return plugin


def test_fork_event_parsing():
    assert ForkEvent.from_slash("fork", "").event_id is None
    assert ForkEvent.from_slash("fork", "abc").event_id == "abc"
    # A leading '@' is stripped.
    assert ForkEvent.from_slash("fork", "@abc").event_id == "abc"


@pytest.mark.asyncio
async def test_handle_fork_success():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event(UserInputEvent(text="hi"))
    ag.add_event(StepEvent())
    e2 = ag.add_event(UserInputEvent(text="again"))
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(event_id=e2.id))
    assert e2.id in msg

    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert e0.id in msg


@pytest.mark.asyncio
async def test_handle_fork_anchor_check():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event(UserInputEvent(text="hi"))
    e1 = ag.add_event(StepEvent())
    plugin = _make_plugin(ag)

    # A user_input event is a valid anchor → succeeds.
    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert e0.id in msg

    # An agent_step event is also a valid anchor now.
    msg = await plugin.handle_fork(ForkEvent(event_id=e1.id))
    assert e1.id in msg


@pytest.mark.asyncio
async def test_handle_fork_missing_or_unknown():
    ag = AgentSession(agent_name="main")
    ag.add_event(UserInputEvent(text="hi"))
    plugin = _make_plugin(ag)

    # No event id supplied.
    msg = await plugin.handle_fork(ForkEvent())
    assert "specify a user turn" in msg

    # Unknown event id.
    msg = await plugin.handle_fork(ForkEvent(event_id="does-not-exist"))
    assert "not found" in msg


@pytest.mark.asyncio
async def test_complete_fork_lists_user_turns_newest_first():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event(UserInputEvent(text="first"))
    ag.add_event(StepEvent())
    e2 = ag.add_event(UserInputEvent(text="second"))
    plugin = _make_plugin(ag)
    # Candidates are now derived from the session's user_input events directly.

    req = CompletionRequest(text="/fork ", cursor=6, current_token="")
    items = [it async for it in plugin.complete_fork(req)]
    assert [it.value for it in items] == [e2.id, e0.id]
    assert [it.label for it in items] == ["@1 (turn 2): second", "@2 (turn 1): first"]
