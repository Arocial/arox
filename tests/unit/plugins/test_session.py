from types import SimpleNamespace

import pytest

from arox.core.completion import CompletionRequest
from arox.core.session import FileSessionStore
from arox.plugins.session import AgentSession, ForkEvent, SessionPlugin
from arox.plugins.slots import AGENT_SESSION
from arox.plugins.subagent import SubagentPlugin


def _make_plugin(main_agent_session: AgentSession):
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
    agent.plugins.append(plugin)

    # The store is normally supplied via SET_SESSION; stub it with a no-op save.
    async def mock_save_session(s):
        pass

    plugin.session_store = SimpleNamespace(save_session=mock_save_session)  # ty: ignore[invalid-assignment]

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


def _make_subagent_plugin(session_id: str | None, owner_path: list[str]):
    """A non-root SessionPlugin wired onto a mock subagent."""

    class MockSubagent:
        def __init__(self):
            self.name = "compaction"
            self.parsed_config = SimpleNamespace(
                app=SimpleNamespace(session_max_age_days=30)
            )
            self.plugins = []
            self.message_history = []
            self.llm_context_id = "ctx"
            self.model_ref = None

        async def reset(self):
            self.message_history = []

        def set_model(self, ref):  # pragma: no cover - model_ref is None here
            pass

        async def invoke_slot(self, slot, *args, **kwargs):
            # Surface the session through the slot, as the real agent does.
            if slot is AGENT_SESSION:
                p = self.get_plugin(SessionPlugin)
                return p.agent_session if p else None
            return []

        def get_plugin(self, plugin_type):
            return next((p for p in self.plugins if isinstance(p, plugin_type)), None)

    agent = MockSubagent()
    plugin = SessionPlugin(agent)
    agent.plugins.append(plugin)
    # The owning SubagentPlugin would feed these in via the SET_SESSION slot.
    plugin.on_set_session(session_id, owner_path, FileSessionStore())
    return plugin


def test_child_session_id_is_stable_and_distinct():
    main_id = "main-session-1"
    a = SubagentPlugin._child_session_id(main_id, "compaction")
    # Deterministic for the same (owner, name)...
    assert a == SubagentPlugin._child_session_id(main_id, "compaction")
    # ...but distinct across owner sessions and across subagent names.
    assert a != SubagentPlugin._child_session_id("main-session-2", "compaction")
    assert a != SubagentPlugin._child_session_id(main_id, "reviewer")


@pytest.mark.asyncio
async def test_subagent_session_nested_under_owner_and_resumes():
    main_id = "main-session-1"
    child_id = SubagentPlugin._child_session_id(main_id, "compaction")

    # First run: the child session is created beneath the owner.
    plugin = _make_subagent_plugin(child_id, [main_id])
    await plugin.on_start()
    session = plugin.agent_session
    assert session is not None
    assert session.id == child_id
    assert session.owner_id == main_id
    assert session.owner_path == [main_id]

    session.add_event("user_input", {"text": "remember me"})
    await plugin.save()

    # Second run with the same derived id/owner resumes the saved session.
    resumed = _make_subagent_plugin(child_id, [main_id])
    await resumed.on_start()
    assert resumed.agent_session is not None
    assert resumed.agent_session.id == child_id
    event_types = [e.event_type for e in resumed.agent_session.events]
    assert "user_input" in event_types


@pytest.mark.asyncio
async def test_fork_reroots_subagent_under_derived_child_id(tmp_path):
    """A forked branch saves its subagents under the id resume re-derives.

    Otherwise the fork-time subagent session would be orphaned and resume would
    silently start from an empty session.
    """
    from arox.core.session import FileSessionStore, derive_child_session_id

    store = FileSessionStore(base_dir=tmp_path / "sessions")

    # A started subagent reachable from the main agent via the SUBAGENTS slot.
    sub_plugin = _make_subagent_plugin(None, [])
    sub_plugin.session_store = store
    await sub_plugin.on_start()
    sub_agent = sub_plugin.agent

    main_session = AgentSession(agent_name="main")
    e0 = main_session.add_event("user_input", {"text": "hi"})

    class MainAgent:
        name = "main"

        def __init__(self):
            self.parsed_config = SimpleNamespace(
                app=SimpleNamespace(session_max_age_days=30)
            )
            self.plugins = []

        async def invoke_slot(self, slot, *args, **kwargs):
            return [sub_agent]

    main_agent = MainAgent()
    main_plugin = SessionPlugin(main_agent)
    main_plugin.agent_session = main_session
    main_plugin.session_store = store
    main_agent.plugins.append(main_plugin)

    msg = await main_plugin.handle_fork(ForkEvent(event_id=e0.id))
    new_branch_id = msg.splitlines()[0].rsplit(":", 1)[-1].strip()

    expected_child = derive_child_session_id(new_branch_id, sub_agent.name)
    loaded = await store.load_session(expected_child, owner_path=[new_branch_id])
    assert loaded is not None
    assert loaded.id == expected_child
    assert loaded.owner_id == new_branch_id
