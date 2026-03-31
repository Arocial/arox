import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from arox.core.session import (
    AgentSession,
    AppSession,
    FileSessionStore,
    _deserialize_messages,
    _serialize_messages,
)


class TestAppSession:
    def test_create(self):
        session = AppSession.create("coder", title="test")
        assert session.composer_name == "coder"
        assert len(session.id) == 12
        assert session.metadata == {"title": "test"}
        assert session.events == []
        assert session.agent_sessions == {}

    def test_add_event(self):
        session = AppSession.create("coder")
        event = session.add_event("system", {"msg": "started"})
        assert event.event_type == "system"
        assert len(session.events) == 1

    def test_get_agent_session(self):
        session = AppSession.create("coder")
        agent_session = session.get_agent_session("main")
        assert agent_session.agent_name == "main"
        assert "main" in session.agent_sessions
        # Second call returns the same instance
        assert session.get_agent_session("main") is agent_session


class TestAgentSession:
    def test_add_event(self):
        agent_session = AgentSession(agent_name="main")
        event = agent_session.add_event("user_input", {"text": "hello"})
        assert event.event_type == "user_input"
        assert event.agent_name == "main"
        assert len(agent_session.events) == 1

    def test_rebuild_empty(self):
        agent_session = AgentSession(agent_name="main")
        examples = [ModelRequest(parts=[UserPromptPart(content="example")])]
        history = agent_session.rebuild_message_history(examples)
        assert len(history) == 1
        assert history[0] is examples[0]

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

        history = agent_session.rebuild_message_history([])
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

        examples = [ModelRequest(parts=[UserPromptPart(content="example")])]
        history = agent_session.rebuild_message_history(examples)
        # example + compacted summary + new step
        assert len(history) == 4
        part0 = history[0].parts[0]
        assert isinstance(part0, UserPromptPart)
        assert part0.content == "example"
        part1 = history[1].parts[0]
        assert isinstance(part1, UserPromptPart)
        assert part1.content == "summary of conversation"
        part2 = history[2].parts[0]
        assert isinstance(part2, UserPromptPart)
        assert part2.content == "new msg"

    def test_non_step_events_ignored_in_rebuild(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event("user_input", {"text": "hello"})
        agent_session.add_event("command", {"command": "/commit"})
        agent_session.add_event("error", {"error": "something"})
        history = agent_session.rebuild_message_history([])
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
        session = AppSession.create("coder")
        agent_session = session.get_agent_session("main")
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
        loaded = await store.load_session(session.id)

        assert loaded is not None
        assert loaded.id == session.id
        assert loaded.composer_name == "coder"

        assert "main" in loaded.agent_sessions
        agent_s = loaded.agent_sessions["main"]
        assert len(agent_s.events) == 2
        assert agent_s.events[0].event_type == "user_input"
        assert agent_s.events[1].event_type == "agent_step"

        # Verify message rebuild works after load
        history = agent_s.rebuild_message_history([])
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, store):
        s1 = AppSession.create("coder")
        s2 = AppSession.create("coder")
        s3 = AppSession.create("other")

        await store.save_session(s1)
        await store.save_session(s2)
        await store.save_session(s3)

        coder_sessions = await store.list_sessions("coder")
        assert len(coder_sessions) == 2
        ids = {s.id for s in coder_sessions}
        assert s1.id in ids
        assert s2.id in ids

        other_sessions = await store.list_sessions("other")
        assert len(other_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, store):
        result = await store.list_sessions("coder")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_session(self, store):
        session = AppSession.create("coder")
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
    async def test_multiple_agent_sessions(self, store):
        session = AppSession.create("coder")
        main_s = session.get_agent_session("main")
        main_s.add_event(
            "agent_step",
            {
                "new_messages": _serialize_messages(
                    [ModelRequest(parts=[UserPromptPart(content="hi")])]
                ),
            },
        )
        comp_s = session.get_agent_session("compaction")
        comp_s.extra = {"some_key": "some_value"}

        await store.save_session(session)
        loaded = await store.load_session(session.id)

        assert loaded is not None
        assert "main" in loaded.agent_sessions
        assert "compaction" in loaded.agent_sessions
        assert len(loaded.agent_sessions["main"].events) == 1
        assert loaded.agent_sessions["compaction"].extra == {"some_key": "some_value"}

    @pytest.mark.asyncio
    async def test_save_overwrites(self, store):
        session = AppSession.create("coder")
        agent_s = session.get_agent_session("main")
        agent_s.add_event("user_input", {"text": "first"})
        await store.save_session(session)

        agent_s.add_event("user_input", {"text": "second"})
        await store.save_session(session)

        loaded = await store.load_session(session.id)
        assert loaded is not None
        assert len(loaded.agent_sessions["main"].events) == 2
