import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from arox.core.session import (
    AgentSession,
    CommandEvent,
    CompactionEvent,
    ErrorEvent,
    FileSessionStore,
    ResetEvent,
    SessionStatus,
    StepEvent,
    UserInputEvent,
)
from arox.core.types import UserInput


class TestAgentSession:
    def test_add_event(self):
        agent_session = AgentSession(agent_name="main")
        event = agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="hello"))
        )
        assert event.event_type == "user_input"
        assert len(agent_session.events) == 1

    def test_persists_only_agent_type(self):
        agent_session = AgentSession(agent_name="main", agent_type="custom")

        data = agent_session.model_dump(mode="json")

        assert data["agent_type"] == "custom"
        assert "agent_config" not in data

    def test_migrates_agent_type_from_legacy_agent_config(self):
        agent_session = AgentSession.model_validate(
            {
                "agent_name": "main",
                "agent_config": {"type": "custom", "system_prompt": "legacy"},
            }
        )

        assert agent_session.agent_type == "custom"
        assert "agent_config" not in agent_session.model_dump(mode="json")

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
        agent_session.add_event(StepEvent(new_messages=messages_step1))
        agent_session.add_event(StepEvent(new_messages=messages_step2))

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
            StepEvent(
                new_messages=[
                    ModelRequest(parts=[UserPromptPart(content="old msg 1")]),
                    ModelResponse(parts=[TextPart(content="old reply 1")]),
                ]
            )
        )
        # Step 2
        agent_session.add_event(
            StepEvent(
                new_messages=[
                    ModelRequest(parts=[UserPromptPart(content="old msg 2")]),
                    ModelResponse(parts=[TextPart(content="old reply 2")]),
                ]
            )
        )
        # Compaction replaces all history
        compacted: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="summary of conversation")])
        ]
        agent_session.add_event(
            CompactionEvent(
                step_boundary=True,
                compacted_messages=compacted,
            )
        )
        # Step 3 after compaction
        agent_session.add_event(
            StepEvent(
                new_messages=[
                    ModelRequest(parts=[UserPromptPart(content="new msg")]),
                    ModelResponse(parts=[TextPart(content="new reply")]),
                ]
            )
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
            CompactionEvent(compacted_messages=[], llm_context_id="ctx_abc123")
        )
        assert agent_session.rebuild_llm_context_id() == "ctx_abc123"

    def test_rebuild_llm_context_id_from_reset(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(
            CompactionEvent(compacted_messages=[], llm_context_id="ctx_first")
        )
        agent_session.add_event(ResetEvent(llm_context_id="ctx_second"))
        assert agent_session.rebuild_llm_context_id() == "ctx_second"

    @pytest.mark.asyncio
    async def test_fork_at_event(self):
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="first"))
        )
        agent_session.add_event(
            StepEvent(
                new_messages=[
                    ModelRequest(parts=[UserPromptPart(content="first")]),
                    ModelResponse(parts=[TextPart(content="r1")]),
                ]
            )
        )
        anchor = agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="second"))
        )

        agent_session.owner = AgentSession(agent_name="parent", path=["parent"])
        forked = await agent_session.fork_at(anchor.id)
        # Independent object truncated just before the anchor event
        assert forked is not agent_session
        assert forked.id != agent_session.id
        assert len(forked.events) == 2
        assert forked.forked_from == (agent_session.path, anchor.id)
        # owner info is correctly inherited
        assert forked.path[:-1] == agent_session.owner.path
        # Original is untouched
        assert len(agent_session.events) == 3

        history = forked.rebuild_message_history()
        assert len(history) == 2
        part = history[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == "first"

    @pytest.mark.asyncio
    async def test_fork_at_none_creates_empty(self):
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        agent_session.owner = AgentSession(agent_name="newowner", path=["newowner"])
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="first"))
        )

        forked = await agent_session.fork_at(None)
        assert forked.events == []
        assert forked.forked_from is None
        # owner taken from the path; a fresh id is minted (located by nesting)
        assert forked.path[:-1] == agent_session.owner.path
        assert forked.id != agent_session.id

    @pytest.mark.asyncio
    async def test_fork_at_missing_event_raises(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="first"))
        )
        with pytest.raises(ValueError):
            agent_session.owner = AgentSession(agent_name="owner", path=["owner"])
            await agent_session.fork_at("does-not-exist")

    @pytest.mark.asyncio
    async def test_fork_at_inherits_owner(self):
        owner = AgentSession(agent_name="parent", path=["parent"])
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        agent_session.owner = owner
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="first"))
        )
        anchor = agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="second"))
        )

        forked = await agent_session.fork_at(anchor.id)
        assert forked.owner is owner
        assert forked.path[:-1] == owner.path
        assert forked.id in owner.children

    def test_non_step_events_ignored_in_rebuild(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="hello"))
        )
        agent_session.add_event(CommandEvent(command="/reset"))
        agent_session.add_event(ErrorEvent(error="something"))
        history = agent_session.rebuild_message_history()
        assert len(history) == 0

    def test_last_user_messages_update(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.record_user_input(
            UserInput(input_content="hello", server_message_id="id1")
        )
        assert agent_session.metadata["last_user_messages"] == ["hello"]

        agent_session.record_user_input(
            UserInput(input_content="world", server_message_id="id2")
        )
        assert agent_session.metadata["last_user_messages"] == ["hello", "world"]

        agent_session.record_user_input(
            UserInput(input_content="third", server_message_id="id3")
        )
        assert agent_session.metadata["last_user_messages"] == ["world", "third"]

        agent_session.record_reset("ctx1")
        assert "last_user_messages" not in agent_session.metadata

    def test_default_status_is_idle(self):
        agent_session = AgentSession(agent_name="main")
        assert agent_session.status == SessionStatus.IDLE
        assert not agent_session.is_active
        assert not agent_session.is_running

    def test_first_class_task_fields(self):
        agent_session = AgentSession(
            agent_name="main",
            task_name="my_task",
            target="/main/my_task",
            initial_message="start",
            last_message="continue",
            last_result="done",
            last_error=None,
        )
        assert agent_session.task_id == agent_session.id
        assert agent_session.task_name == "my_task"
        assert agent_session.target == "/main/my_task"
        assert agent_session.result == "done"
        assert agent_session.error is None

    @pytest.mark.asyncio
    async def test_runtime_and_running_task_excluded_from_serialization(self):
        agent_session = AgentSession(agent_name="main")
        dummy_task = asyncio.create_task(asyncio.sleep(0))
        agent_session.runtime = "dummy_runtime"
        agent_session.running_task = dummy_task

        dumped = agent_session.model_dump(mode="json")
        assert "runtime" not in dumped
        assert "running_task" not in dumped
        assert agent_session.agent == "dummy_runtime"
        await dummy_task

    def test_status_and_result_helpers(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.status = SessionStatus.ACTIVE
        assert agent_session.is_active

        agent_session.record_result("all done")
        assert agent_session.status == SessionStatus.COMPLETED
        assert agent_session.last_result == "all done"
        assert agent_session.last_error is None

        agent_session.record_interrupted("cancelled by user")
        assert agent_session.status == SessionStatus.INTERRUPTED
        assert agent_session.last_error == "cancelled by user"

        agent_session.record_error("something crashed")
        assert agent_session.status == SessionStatus.ERRORED
        assert agent_session.last_error == "something crashed"
        assert agent_session.events[-1].event_type == "error"

        agent_session.close_session()
        assert agent_session.status == SessionStatus.CLOSED
        assert agent_session.runtime is None

    def test_legacy_subagent_task_migration(self):
        data = {
            "agent_name": "planner",
            "extra": {
                "subagent_task": {
                    "task_id": "task_123",
                    "task_name": "plan_feature",
                    "target": "/main/plan_feature",
                    "agent_type": "planner",
                    "initial_message": "plan x",
                    "last_message": "plan x",
                    "result": "the plan",
                    "error": None,
                    "status": "completed",
                }
            },
        }
        session = AgentSession.model_validate(data)
        assert session.task_name == "plan_feature"
        assert session.target == "/main/plan_feature"
        assert session.initial_message == "plan x"
        assert session.last_message == "plan x"
        assert session.last_result == "the plan"
        assert session.status == SessionStatus.COMPLETED
        assert "subagent_task" not in session.extra

    @pytest.mark.asyncio
    async def test_fork_resets_status_and_task_fields(self):
        original = AgentSession(
            agent_name="main",
            status=SessionStatus.COMPLETED,
            last_result="res",
            last_error="err",
        )
        original.add_event(UserInputEvent(user_input=UserInput(input_content="hi")))
        forked = await original.fork_at(None)
        assert forked.status == SessionStatus.IDLE
        assert forked.last_result is None
        assert forked.last_error is None


class TestFileSessionStore:
    @pytest.fixture
    def store(self, tmp_path):
        from arox.core.session import AgentSession, SessionManager

        s = FileSessionStore()
        s.base_dir = tmp_path / "sessions"
        sm = SessionManager(s)
        sm.register_session_type(AgentSession)
        return s

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        # A top-level main-agent session with a subagent session nested under it.
        session = AgentSession(agent_name="coder", path=["coder"])
        agent_session = AgentSession(agent_name="sub", path=["coder", "sub"])
        agent_session.add_event(
            UserInputEvent(user_input=UserInput(input_content="hello"))
        )
        agent_session.add_event(
            StepEvent(
                new_messages=[
                    ModelRequest(parts=[UserPromptPart(content="hello")]),
                    ModelResponse(parts=[TextPart(content="hi there")]),
                ]
            )
        )

        await store.save_session(session)
        await store.save_session(agent_session)

        loaded = await store.load_session(session.path)
        assert loaded is not None
        assert loaded.id == session.id
        assert isinstance(loaded, AgentSession)
        assert loaded.agent_name == "coder"

        loaded_agent = await store.load_session(agent_session.path)
        assert loaded_agent is not None
        assert isinstance(loaded_agent, AgentSession)
        assert loaded_agent.agent_name == "sub"
        assert len(loaded_agent.events) == 2
        assert loaded_agent.events[0].event_type == "user_input"
        assert loaded_agent.events[1].event_type == "agent_step"

    @pytest.mark.asyncio
    async def test_load_keeps_children_as_ids(self, store):
        # main -> sub -> grandchild nesting, each persisted under its owner.
        main = AgentSession(agent_name="main", path=["main"])
        sub = AgentSession(agent_name="sub", path=["main", "sub"])
        grand = AgentSession(agent_name="grand", path=["main", "sub", "grand"])
        main.children.append(sub.id)
        sub.children.append(grand.id)

        await store.save_session(main)
        await store.save_session(sub)
        await store.save_session(grand)

        loaded = await store.load_session(main.path)
        assert loaded is not None
        # Subsessions are referenced by id and loaded explicitly by callers.
        assert loaded.children == [sub.id]
        loaded_sub = await store.load_session(sub.path)
        assert isinstance(loaded_sub, AgentSession)
        assert loaded_sub.children == [grand.id]

    @pytest.mark.asyncio
    async def test_children_persisted_as_ids(self, store):
        main = AgentSession(agent_name="main", path=["main"])
        sub = AgentSession(agent_name="sub", path=["main", "sub"])
        main.children.append(sub.id)
        await store.save_session(main)
        await store.save_session(sub)

        import json

        raw = json.loads(store._session_meta_path(main.path).read_text())
        assert "children" in raw
        assert raw["children"] == [sub.id]

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load_session(["nonexistent"])
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
        session = AgentSession(agent_name="coder", path=["coder"])
        await store.save_session(session)

        loaded = await store.load_session(session.path)
        assert loaded is not None

        await store.delete_session(session.path)
        loaded = await store.load_session(session.path)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        await store.delete_session(["nonexistent"])

    @pytest.mark.asyncio
    async def test_save_overwrites(self, store):
        agent_s = AgentSession(agent_name="main", path=["coder", "main"])
        agent_s.add_event(UserInputEvent(user_input=UserInput(input_content="first")))
        await store.save_session(agent_s)

        agent_s.add_event(UserInputEvent(user_input=UserInput(input_content="second")))
        await store.save_session(agent_s)

        loaded = await store.load_session(agent_s.path)
        assert loaded is not None
        assert len(loaded.events) == 2

    def _backdate_session(self, store, session, days):
        """Save session then overwrite updated_at to simulate an old session."""
        import json

        meta_path = store._session_meta_path(session.path)
        raw = json.loads(meta_path.read_text())
        raw["updated_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        meta_path.write_text(json.dumps(raw))

    @pytest.mark.asyncio
    async def test_cleanup_deletes_expired(self, store):
        old_session = AgentSession(agent_name="coder", path=["old"])
        await store.save_session(old_session)
        self._backdate_session(store, old_session, days=60)

        new_session = AgentSession(agent_name="coder", path=["new"])
        await store.save_session(new_session)

        deleted = await store.cleanup(max_age_days=30)
        assert deleted == 1

        assert await store.load_session(old_session.path) is None
        assert await store.load_session(new_session.path) is not None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent(self, store):
        session = AgentSession(agent_name="coder", path=["recent"])
        await store.save_session(session)

        deleted = await store.cleanup(max_age_days=30)
        assert deleted == 0
        assert await store.load_session(session.path) is not None

    @pytest.mark.asyncio
    async def test_cleanup_empty_store(self, store):
        deleted = await store.cleanup()
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_uses_default_max_age(self, tmp_path):
        store = FileSessionStore(max_age_days=7)
        store.base_dir = tmp_path / "sessions"
        old_session = AgentSession(agent_name="coder")
        await store.save_session(old_session)
        self._backdate_session(store, old_session, days=10)

        deleted = await store.cleanup()
        assert deleted == 1
