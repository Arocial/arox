from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from arox.core.session import (
    FileSessionStore,
    _deserialize_messages,
    _serialize_messages,
)
from arox.plugins.session import AgentSession


class TestAgentSession:
    def test_add_event(self):
        agent_session = AgentSession(agent_name="main")
        event = agent_session.add_event("user_input", {"text": "hello"})
        assert event.event_type == "user_input"
        assert len(agent_session.events) == 1

    def test_rebuild_empty(self):
        agent_session = AgentSession(agent_name="main")
        history = agent_session.rebuild_message_history()
        assert history == []

    def test_rebuild_from_steps(self):
        agent_session = AgentSession(agent_name="main")
        messages_step1 = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="hi")]),
        ]
        messages_step2 = [
            ModelRequest(parts=[UserPromptPart(content="bye")]),
            ModelResponse(parts=[TextPart(content="goodbye")]),
        ]
        agent_session.add_event(
            "agent_step",
            {"new_messages": _serialize_messages(messages_step1)},
        )
        agent_session.add_event(
            "agent_step",
            {"new_messages": _serialize_messages(messages_step2)},
        )

        history = agent_session.rebuild_message_history()
        assert len(history) == 4
        assert isinstance(history[0], ModelRequest)
        part = history[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == "hello"

    def test_rebuild_with_compaction(self):
        agent_session = AgentSession(agent_name="main")
        # Step 1
        agent_session.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [
                        ModelRequest(parts=[UserPromptPart(content="old msg 1")]),
                        ModelResponse(parts=[TextPart(content="old reply 1")]),
                    ]
                ),
            },
        )
        # Step 2
        agent_session.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [
                        ModelRequest(parts=[UserPromptPart(content="old msg 2")]),
                        ModelResponse(parts=[TextPart(content="old reply 2")]),
                    ]
                ),
            },
        )
        # Compaction replaces all history
        compacted = [
            ModelRequest(parts=[UserPromptPart(content="summary of conversation")])
        ]
        agent_session.add_event(
            "compaction",
            {"compacted_messages": _serialize_messages(compacted)},
        )
        # Step 3 after compaction
        agent_session.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [
                        ModelRequest(parts=[UserPromptPart(content="new msg")]),
                        ModelResponse(parts=[TextPart(content="new reply")]),
                    ]
                ),
            },
        )

        history = agent_session.rebuild_message_history()
        # compacted summary + new step
        assert len(history) == 3
        part0 = history[0].parts[0]
        assert isinstance(part0, UserPromptPart)
        assert part0.content == "summary of conversation"
        part1 = history[1].parts[0]
        assert isinstance(part1, UserPromptPart)
        assert part1.content == "new msg"

    def test_rebuild_llm_context_id_none_without_events(self):
        agent_session = AgentSession(agent_name="main")
        assert agent_session.rebuild_llm_context_id() is None

    def test_rebuild_llm_context_id_from_compaction(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(
            "compaction",
            {"compacted_messages": [], "llm_context_id": "ctx_abc123"},
        )
        assert agent_session.rebuild_llm_context_id() == "ctx_abc123"

    def test_rebuild_llm_context_id_from_reset(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(
            "compaction",
            {"compacted_messages": [], "llm_context_id": "ctx_first"},
        )
        agent_session.add_event(
            "reset",
            {"llm_context_id": "ctx_second"},
        )
        assert agent_session.rebuild_llm_context_id() == "ctx_second"

    def test_fork_at_event(self):
        agent_session = AgentSession(agent_name="main", owner_id="parent")
        agent_session.add_event("user_input", {"text": "first"})
        agent_session.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [
                        ModelRequest(parts=[UserPromptPart(content="first")]),
                        ModelResponse(parts=[TextPart(content="r1")]),
                    ]
                ),
            },
        )
        anchor = agent_session.add_event("user_input", {"text": "second"})

        forked = agent_session.fork_at(anchor.id, [])
        # Independent object truncated just before the anchor event
        assert forked is not agent_session
        assert forked.id != agent_session.id
        assert len(forked.events) == 2
        assert forked.forked_from == {"main": 2}
        # owner info is corrected to the (empty) owner path, not inherited
        assert forked.owner_id is None
        assert forked.owner_path == []
        # Original is untouched
        assert len(agent_session.events) == 3

        history = forked.rebuild_message_history()
        assert len(history) == 2
        part = history[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == "first"

    def test_fork_at_none_creates_empty(self):
        from arox.core.session import derive_child_session_id

        agent_session = AgentSession(agent_name="main", owner_id="parent")
        agent_session.add_event("user_input", {"text": "first"})

        forked = agent_session.fork_at(None, ["newowner"])
        assert forked.events == []
        assert forked.forked_from is None
        # owner taken from the path, id derived so the owner can re-find it
        assert forked.owner_id == "newowner"
        assert forked.owner_path == ["newowner"]
        assert forked.id == derive_child_session_id("newowner", "main")

    def test_fork_at_missing_event_raises(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event("user_input", {"text": "first"})
        with pytest.raises(ValueError):
            agent_session.fork_at("does-not-exist", [])

    def test_non_step_events_ignored_in_rebuild(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event("user_input", {"text": "hello"})
        agent_session.add_event("command", {"command": "/reset"})
        agent_session.add_event("error", {"error": "something"})
        history = agent_session.rebuild_message_history()
        assert len(history) == 0


class TestMessageSerialization:
    def test_round_trip(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="world")]),
        ]
        serialized = _serialize_messages(messages)
        assert isinstance(serialized, list)
        assert len(serialized) == 2

        deserialized = _deserialize_messages(serialized)
        assert len(deserialized) == 2
        assert isinstance(deserialized[0], ModelRequest)
        assert isinstance(deserialized[1], ModelResponse)
        part0 = deserialized[0].parts[0]
        assert isinstance(part0, UserPromptPart)
        assert part0.content == "hello"
        part1 = deserialized[1].parts[0]
        assert isinstance(part1, TextPart)
        assert part1.content == "world"

    def test_empty(self):
        assert _serialize_messages([]) == []
        assert _deserialize_messages([]) == []


class TestFileSessionStore:
    @pytest.fixture
    def store(self, tmp_path):
        return FileSessionStore(base_dir=tmp_path / "sessions")

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        # A top-level main-agent session with a subagent session nested under it.
        session = AgentSession(agent_name="coder")
        agent_session = AgentSession(
            agent_name="sub", owner_id=session.id, owner_path=[session.id]
        )
        agent_session.add_event("user_input", {"text": "hello"})
        agent_session.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [
                        ModelRequest(parts=[UserPromptPart(content="hello")]),
                        ModelResponse(parts=[TextPart(content="hi there")]),
                    ]
                ),
            },
        )

        await store.save_session(session)
        await store.save_session(agent_session)

        loaded = await store.load_session(session.id)
        assert loaded is not None
        assert loaded.id == session.id
        assert isinstance(loaded, AgentSession)
        assert loaded.agent_name == "coder"

        loaded_agent = await store.load_session(
            agent_session.id, owner_path=[session.id]
        )
        assert loaded_agent is not None
        assert isinstance(loaded_agent, AgentSession)
        assert loaded_agent.agent_name == "sub"
        assert len(loaded_agent.events) == 2
        assert loaded_agent.events[0].event_type == "user_input"
        assert loaded_agent.events[1].event_type == "agent_step"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, store):
        s1 = AgentSession(agent_name="coder")
        s2 = AgentSession(agent_name="coder")
        s3 = AgentSession(agent_name="other")

        await store.save_session(s1)
        await store.save_session(s2)
        await store.save_session(s3)

        sessions = await store.list_sessions()
        assert len(sessions) == 3
        # The store lists all agent sessions; filtering by agent_name is a
        # caller concern.
        coder_sessions = [
            s
            for s in sessions
            if isinstance(s, AgentSession) and s.agent_name == "coder"
        ]
        assert len(coder_sessions) == 2
        ids = {s.id for s in coder_sessions}
        assert s1.id in ids
        assert s2.id in ids

        other_sessions = [
            s
            for s in sessions
            if isinstance(s, AgentSession) and s.agent_name == "other"
        ]
        assert len(other_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, store):
        result = await store.list_sessions()
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_session(self, store):
        session = AgentSession(agent_name="coder")
        await store.save_session(session)

        loaded = await store.load_session(session.id)
        assert loaded is not None

        await store.delete_session(session.id)
        loaded = await store.load_session(session.id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        await store.delete_session("nonexistent")

    @pytest.mark.asyncio
    async def test_save_overwrites(self, store):
        session = AgentSession(agent_name="coder")
        agent_s = AgentSession(
            agent_name="main", owner_id=session.id, owner_path=[session.id]
        )
        agent_s.add_event("user_input", {"text": "first"})
        await store.save_session(agent_s)

        agent_s.add_event("user_input", {"text": "second"})
        await store.save_session(agent_s)

        loaded = await store.load_session(agent_s.id, owner_path=[session.id])
        assert loaded is not None
        assert len(loaded.events) == 2

    def _backdate_session(self, store, session, days):
        """Save session then overwrite updated_at to simulate an old session."""
        import json

        meta_path = store._session_meta_path(session.id)
        raw = json.loads(meta_path.read_text())
        raw["updated_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        meta_path.write_text(json.dumps(raw))

    @pytest.mark.asyncio
    async def test_cleanup_deletes_expired(self, store):
        old_session = AgentSession(agent_name="coder")
        await store.save_session(old_session)
        self._backdate_session(store, old_session, days=60)

        new_session = AgentSession(agent_name="coder")
        await store.save_session(new_session)

        deleted = await store.cleanup(max_age_days=30)
        assert deleted == 1

        assert await store.load_session(old_session.id) is None
        assert await store.load_session(new_session.id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent(self, store):
        session = AgentSession(agent_name="coder")
        await store.save_session(session)

        deleted = await store.cleanup(max_age_days=30)
        assert deleted == 0
        assert await store.load_session(session.id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_empty_store(self, store):
        deleted = await store.cleanup()
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_uses_default_max_age(self, tmp_path):
        store = FileSessionStore(base_dir=tmp_path / "sessions", max_age_days=7)
        old_session = AgentSession(agent_name="coder")
        await store.save_session(old_session)
        self._backdate_session(store, old_session, days=10)

        deleted = await store.cleanup()
        assert deleted == 1
