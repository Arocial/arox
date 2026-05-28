from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.completion import CompletionRequest
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
    assert ForkEvent.from_slash("fork", "").event_id is None
    assert ForkEvent.from_slash("fork", "abc").event_id == "abc"
    # A leading '@' is stripped.
    assert ForkEvent.from_slash("fork", "@abc").event_id == "abc"


@pytest.mark.asyncio
async def test_handle_fork_success():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    e2 = ag.add_event("user_input", {"text": "again"})
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(event_id=e2.id))
    assert "Forked at event @2" in msg

    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert "Forked at event @0" in msg


@pytest.mark.asyncio
async def test_handle_fork_anchor_check():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event("user_input", {"text": "hi"})
    e1 = ag.add_event("agent_step", {})
    plugin = _make_plugin(ag)

    # A user_input event is a valid anchor → succeeds.
    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert "Forked at event @0" in msg

    # An agent_step event is not an anchor → rejected.
    msg = await plugin.handle_fork(ForkEvent(event_id=e1.id))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_fork_missing_or_unknown():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    plugin = _make_plugin(ag)

    # No event id supplied.
    msg = await plugin.handle_fork(ForkEvent())
    assert "specify a user turn" in msg

    # Unknown event id.
    msg = await plugin.handle_fork(ForkEvent(event_id="does-not-exist"))
    assert "event not found" in msg


def _tagged_user_message(text: str, input_id: str):
    from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

    from arox.core.session import USER_INPUT_ID_KEY

    return ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content=text + "\n", metadata={USER_INPUT_ID_KEY: input_id}
                    )
                ]
            )
        ]
    )


@pytest.mark.asyncio
async def test_complete_fork_lists_user_turns_newest_first():
    ag = AgentSession(agent_name="main")
    e0 = ag.add_event("user_input", {"text": "first"})
    ag.add_event("agent_step", {})
    e2 = ag.add_event("user_input", {"text": "second"})
    plugin = _make_plugin(ag)
    # Candidates are derived from the live message history's USER_INPUT_ID_KEY
    # metadata, not the user_input events directly.
    plugin.agent.message_history = [
        _tagged_user_message("first", e0.id),
        _tagged_user_message("second", e2.id),
    ]

    req = CompletionRequest(text="/fork ", cursor=6, current_token="")
    items = [it async for it in plugin.complete_fork(req)]
    assert [it.value for it in items] == [e2.id, e0.id]
    assert [it.label for it in items] == ["@1 (turn 2): second", "@2 (turn 1): first"]
