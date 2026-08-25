from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from arox.core.completion import CompletionRequest
from arox.core.config import AgentConfig
from arox.core.session import AgentSession, StepEvent, UserInputEvent
from arox.core.types import ClientInput, MessagePayload, normalize_client_input
from arox.plugins.core import CorePlugin, ForkEvent


def _message_input(content):
    return normalize_client_input(ClientInput(payload=MessagePayload(content=content)))


def _user_turn(text: str) -> tuple[UserInputEvent, ModelRequest]:
    user_input = _message_input(text)
    payload = user_input.payload
    assert isinstance(payload, MessagePayload)
    assert payload.content is not None
    assert user_input.server_message_id is not None
    event = UserInputEvent(id=user_input.server_message_id, client_input=user_input)
    request = ModelRequest(parts=[UserPromptPart(content=payload.content)])
    return event, request


def _make_plugin(main_agent_session: AgentSession):
    class MockAgent:
        session: AgentSession | None = None
        owner: AgentSession | None = None

        def __init__(self):
            self.name = main_agent_session.agent_name
            self.agent_config = AgentConfig()
            self.config = SimpleNamespace(app=SimpleNamespace(session_max_age_days=30))
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
    e0, request0 = _user_turn("hi")
    ag.add_event(e0)
    response0 = ModelResponse(parts=[TextPart(content="hello")])
    ag.record_step(
        [request0, response0],
        input_event_id=e0.id,
        new_messages=[request0, response0],
    )
    e2, request2 = _user_turn("again")
    ag.add_event(e2)
    response2 = ModelResponse(parts=[TextPart(content="ok")])
    ag.record_step(
        [request0, response0, request2, response2],
        input_event_id=e2.id,
        new_messages=[request2, response2],
    )
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(event_id=e2.id))
    assert e2.id in msg

    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert e0.id in msg


@pytest.mark.asyncio
async def test_handle_fork_requires_user_input_event():
    ag = AgentSession(agent_name="main")
    e0, request = _user_turn("hi")
    ag.add_event(e0)
    response = ModelResponse(parts=[TextPart(content="hello")])
    ag.record_step(
        [request, response],
        input_event_id=e0.id,
        new_messages=[request, response],
    )
    e1 = ag.events[-1]
    plugin = _make_plugin(ag)

    msg = await plugin.handle_fork(ForkEvent(event_id=e0.id))
    assert e0.id in msg

    msg = await plugin.handle_fork(ForkEvent(event_id=e1.id))
    assert msg == f"event {e1.id} is not a user input"


@pytest.mark.asyncio
async def test_handle_fork_missing_or_unknown():
    ag = AgentSession(agent_name="main")
    ag.add_event(UserInputEvent(client_input=_message_input("hi")))
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
    e0 = ag.add_event(UserInputEvent(client_input=_message_input("first")))
    ag.add_event(StepEvent())
    e2 = ag.add_event(UserInputEvent(client_input=_message_input("second")))
    plugin = _make_plugin(ag)
    # Candidates are now derived from the session's user_input events directly.

    req = CompletionRequest(text="/fork ", cursor=6, current_token="")
    items = [it async for it in plugin.complete_fork(req)]
    assert [it.value for it in items] == [e2.id, e0.id]
    assert [it.label for it in items] == ["@1 (turn 2): second", "@2 (turn 1): first"]
