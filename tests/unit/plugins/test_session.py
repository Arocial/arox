from types import SimpleNamespace

import pytest

from arox.core.completion import CompletionRequest
from arox.core.session import FileSessionStore
from arox.plugins.session import AgentSession, ForkEvent, SessionPlugin
from arox.plugins.slots import AGENT_SESSION


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


def _make_subagent_plugin():
    """A SessionPlugin wired onto a mock subagent (session not yet resolved)."""

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
    return plugin


@pytest.mark.asyncio
async def test_subagent_session_nests_under_owner_and_resumes(tmp_path):
    store = FileSessionStore(base_dir=tmp_path / "sessions")
    main_id = "main-session-1"

    # First run: a fresh child session is created nested beneath the owner.
    plugin = _make_subagent_plugin()
    await plugin.on_set_session(None, [main_id], store)
    session = plugin.agent_session
    assert session is not None
    assert session.owner_id == main_id
    assert session.owner_path == [main_id]

    session.add_event("user_input", {"text": "remember me"})
    await plugin.save()

    # Persisted nested under the owner, so the owner's recursive load finds it.
    reloaded = await store.load_session(session.id, owner_path=[main_id])
    assert reloaded is not None
    assert [e.event_type for e in reloaded.events] == ["user_input"]

    # Second run: the owner hands the loaded session back as a preset.
    resumed = _make_subagent_plugin()
    await resumed.on_set_session(None, [main_id], store, reloaded)
    assert resumed.agent_session is reloaded
    assert [e.event_type for e in resumed.agent_session.events] == ["user_input"]


@pytest.mark.asyncio
async def test_set_session_adopts_preset_without_loading():
    # A subsession the owner already loaded off the session tree.
    preset = AgentSession(agent_name="compaction", owner_id="main", owner_path=["main"])
    preset.add_event("user_input", {"text": "remembered"})

    plugin = _make_subagent_plugin()

    async def fail(*args, **kwargs):
        raise AssertionError("the store must not be touched when a preset is given")

    store = SimpleNamespace(load_session=fail, cleanup=fail)
    await plugin.on_set_session(None, ["main"], store, preset)

    # Adopted as-is and its history rebuilt; the store was never queried.
    assert plugin.agent_session is preset
    assert plugin.agent.message_history == preset.rebuild_message_history()


@pytest.mark.asyncio
async def test_fork_reroots_subagents_under_new_branch(tmp_path):
    """A forked branch re-roots its subsessions beneath the new branch.

    Otherwise the fork-time subagent session would be orphaned and resume would
    silently start from an empty session.
    """
    store = FileSessionStore(base_dir=tmp_path / "sessions")

    # A started subagent whose live session is linked under the main session.
    sub_plugin = _make_subagent_plugin()
    await sub_plugin.on_set_session(None, ["owner"], store)
    sub_agent = sub_plugin.agent

    main_session = AgentSession(agent_name="main")
    e0 = main_session.add_event("user_input", {"text": "hi"})
    # SubagentPlugin would link the live subagent session here; the fork reads
    # subsessions from children rather than routing through the live subagent.
    main_session.children.append(sub_plugin.agent_session)

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

    # The new branch, loaded recursively, carries the re-rooted subsession.
    loaded = await store.load_session(new_branch_id)
    assert isinstance(loaded, AgentSession)
    assert [c.agent_name for c in loaded.children] == [sub_agent.name]
    child = loaded.children[0]
    assert child.owner_id == new_branch_id
    assert child.owner_path == [new_branch_id]
