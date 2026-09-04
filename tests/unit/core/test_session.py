import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from arox.core.agent_runtime import AgentRuntime
from arox.core.app import app_setup
from arox.core.io import AbstractIOAdapter
from arox.core.session import (
    AgentSession,
    CommandCompletedEvent,
    ErrorEvent,
    FileSessionStore,
    ModelMessageEvent,
    SessionManager,
    UserInputEvent,
)
from arox.core.types import (
    ClientInput,
    CommandPayload,
    MessagePayload,
    normalize_client_input,
)
from tests.history import (
    compact_history,
    contains_input,
    context_resets,
    record_messages,
    reset_history,
)


def _message_input(content, **kwargs):
    return normalize_client_input(
        ClientInput(payload=MessagePayload(content=content), **kwargs)
    )


def _command_input(command, **kwargs):
    return normalize_client_input(
        ClientInput(payload=CommandPayload(command=command), **kwargs)
    )


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_ep, event):
        pass


def _user_turn(text: str) -> tuple[UserInputEvent, ModelRequest]:
    user_input = _message_input(text)
    payload = user_input.payload
    assert isinstance(payload, MessagePayload)
    assert payload.content is not None
    assert user_input.server_message_id is not None
    event = UserInputEvent(id=user_input.server_message_id, client_input=user_input)
    request = ModelRequest(parts=[UserPromptPart(content=payload.content)])
    return event, request


def _record_step(
    session: AgentSession,
    messages: list[ModelMessage],
) -> None:
    previous = {id(message) for message in session.message_history}
    new_messages = [message for message in messages if id(message) not in previous]
    run_id = uuid.uuid4().hex
    for sequence, message in enumerate(new_messages):
        session.record_model_message(message, run_id=run_id, sequence=sequence)


class TestAgentSession:
    def test_add_event(self):
        agent_session = AgentSession(agent_name="main")
        event = agent_session.add_event(
            UserInputEvent(client_input=_message_input("hello"))
        )
        assert event.event_type == "user_input"
        assert len(agent_session.journal) == 1

    def test_runtime_is_not_persisted(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.runtime = object()

        data = agent_session.model_dump(mode="json")

        assert "runtime" not in data

    def test_message_history_defaults_empty_and_is_serialized(self):
        agent_session = AgentSession(agent_name="main")
        assert agent_session.message_history == []
        assert context_resets(agent_session) == []

        dumped = agent_session.model_dump(mode="json")
        assert dumped["journal"] == []
        assert "context_resets" not in dumped

    def test_user_input_ids_are_read_from_message_content(self):
        agent_session = AgentSession(agent_name="main")
        first_input, first_request = _user_turn("hello")
        agent_session.add_event(first_input)
        messages_step1 = [
            first_request,
            ModelResponse(parts=[TextPart(content="hi")]),
        ]
        _record_step(agent_session, messages_step1)
        second_input, second_request = _user_turn("bye")
        agent_session.add_event(second_input)
        messages_step2 = [
            *messages_step1,
            second_request,
            ModelResponse(parts=[TextPart(content="goodbye")]),
        ]
        _record_step(agent_session, messages_step2)

        history = agent_session.message_history
        assert history == messages_step2
        assert contains_input(history, first_input.id)
        assert contains_input(history, second_input.id)

    @pytest.mark.asyncio
    async def test_fork_preserves_journal_messages_before_user_input(self):
        agent_session = AgentSession(agent_name="main")
        first_input, first_request = _user_turn("first")
        agent_session.add_event(first_input)
        response = ModelResponse(parts=[TextPart(content="reply")])
        _record_step(agent_session, [first_request, response])
        inserted = ModelRequest(parts=[UserPromptPart(content="inserted context")])
        agent_session.record_model_message(inserted, run_id="injected", sequence=0)
        second_input, second_request = _user_turn("second")
        agent_session.add_event(second_input)
        _record_step(
            agent_session,
            [
                first_request,
                response,
                second_request,
                ModelResponse(parts=[TextPart(content="second reply")]),
            ],
        )

        forked = await agent_session.fork_at(second_input.id)

        history = forked.message_history
        assert len(history) == 3
        assert history[1].parts == response.parts
        assert history[2].parts == inserted.parts
        assert not contains_input(history, second_input.id)

    def test_compaction_resets_context_without_removing_journal_messages(self):
        agent_session = AgentSession(agent_name="main")
        old_input, old_request = _user_turn("old msg")
        agent_session.add_event(old_input)
        old_messages = [
            old_request,
            ModelResponse(parts=[TextPart(content="old reply")]),
        ]
        _record_step(agent_session, old_messages)

        compacted: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="summary of conversation")])
        ]
        compact_history(agent_session, compacted, True, "ctx-summary")

        new_messages = [
            *compacted,
            ModelRequest(parts=[UserPromptPart(content="new msg")]),
            ModelResponse(parts=[TextPart(content="new reply")]),
        ]
        _record_step(agent_session, new_messages)

        assert len(context_resets(agent_session)) == 1
        reset = context_resets(agent_session)[0]
        reset_index = agent_session.index_of_event(reset.id)
        assert reset_index is not None
        summary_event = agent_session.journal[reset_index + 1]
        assert isinstance(summary_event, ModelMessageEvent)
        assert summary_event.context_only
        assert summary_event.message == compacted[0]
        assert "messages" not in reset.model_dump()
        assert agent_session.message_history == new_messages
        assert agent_session.run_info.llm_context_id == "ctx-summary"

    @pytest.mark.asyncio
    async def test_fork_at_event(self):
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        first_input, first_request = _user_turn("first")
        agent_session.add_event(first_input)
        first_messages = [
            first_request,
            ModelResponse(parts=[TextPart(content="r1")]),
        ]
        _record_step(agent_session, first_messages)
        anchor, second_request = _user_turn("second")
        agent_session.add_event(anchor)
        _record_step(
            agent_session,
            [
                *first_messages,
                second_request,
                ModelResponse(parts=[TextPart(content="r2")]),
            ],
        )

        agent_session.owner = AgentSession(agent_name="parent", path=["parent"])
        forked = await agent_session.fork_at(anchor.id)
        # Independent object truncated just before the anchor event
        assert forked is not agent_session
        assert forked.id != agent_session.id
        assert len(forked.journal) == 3
        assert forked.forked_from == (agent_session.path, anchor.id)
        # owner info is correctly inherited
        assert forked.path[:-1] == agent_session.owner.path
        # Original is untouched
        assert len(agent_session.journal) == 6

        assert len(forked.message_history) == 2
        part = forked.message_history[0].parts[0]
        assert isinstance(part, UserPromptPart)
        payload = first_input.client_input.payload
        assert isinstance(payload, MessagePayload)
        assert part.content == payload.content

    @pytest.mark.asyncio
    async def test_fork_uses_archived_history_after_compaction(self):
        agent_session = AgentSession(agent_name="main")
        first_input, first_request = _user_turn("first")
        agent_session.add_event(first_input)
        first_messages = [
            first_request,
            ModelResponse(parts=[TextPart(content="r1")]),
        ]
        _record_step(agent_session, first_messages)
        second_input, second_request = _user_turn("second")
        agent_session.add_event(second_input)

        compacted = [
            ModelRequest(parts=[UserPromptPart(content="summary including second")])
        ]
        compact_history(agent_session, compacted, False, "ctx-compact")
        _record_step(
            agent_session, [*compacted, ModelResponse(parts=[TextPart(content="r2")])]
        )

        forked = await agent_session.fork_at(second_input.id)

        history = forked.message_history
        assert history == first_messages
        assert contains_input(history, first_input.id)
        assert not contains_input(history, second_input.id)

    @pytest.mark.asyncio
    async def test_fork_at_none_creates_empty(self):
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        agent_session.owner = AgentSession(agent_name="newowner", path=["newowner"])
        agent_session.add_event(UserInputEvent(client_input=_message_input("first")))

        forked = await agent_session.fork_at(None)
        assert forked.journal == []
        assert forked.forked_from is None
        # owner taken from the path; a fresh id is minted (located by nesting)
        assert forked.path[:-1] == agent_session.owner.path
        assert forked.id != agent_session.id

    @pytest.mark.asyncio
    async def test_fork_does_not_copy_active_runtime(self):
        runtime = SimpleNamespace(lock=RLock())
        agent_session = AgentSession(agent_name="main", runtime=runtime)

        forked = await agent_session.fork_at(None)

        assert agent_session.runtime is runtime
        assert forked.runtime is None

    @pytest.mark.asyncio
    async def test_fork_at_missing_event_raises(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(UserInputEvent(client_input=_message_input("first")))
        with pytest.raises(ValueError):
            agent_session.owner = AgentSession(agent_name="owner", path=["owner"])
            await agent_session.fork_at("does-not-exist")

    @pytest.mark.asyncio
    async def test_fork_at_inherits_owner(self):
        owner = AgentSession(agent_name="parent", path=["parent"])
        agent_session = AgentSession(agent_name="main", path=["parent", "main"])
        agent_session.owner = owner
        first_input, first_request = _user_turn("first")
        agent_session.add_event(first_input)
        first_response = ModelResponse(parts=[TextPart(content="r1")])
        _record_step(agent_session, [first_request, first_response])
        anchor, second_request = _user_turn("second")
        agent_session.add_event(anchor)
        _record_step(
            agent_session,
            [
                first_request,
                first_response,
                second_request,
                ModelResponse(parts=[TextPart(content="r2")]),
            ],
        )

        forked = await agent_session.fork_at(anchor.id)
        assert forked.owner is owner
        assert forked.path[:-1] == owner.path
        assert forked.id in owner.children

    def test_io_timeline_interleaves_commands_with_model_turns(self):
        agent_session = AgentSession(agent_name="main")
        user_event, request = _user_turn("analyze")
        response = ModelResponse(parts=[TextPart(content="analysis complete")])
        agent_session.add_event(user_event)

        agent_session.record_command_completed(
            _command_input("/info"),
            "handled",
            output="model details",
        )
        record_messages(agent_session, [request, response])

        assert context_resets(agent_session) == []
        snapshot = agent_session.build_io_timeline()
        assert len(snapshot) == 3
        assert isinstance(snapshot[0], ModelRequest)
        payload = user_event.client_input.payload
        assert isinstance(payload, MessagePayload)
        assert snapshot[0].parts[0].content == payload.content
        assert isinstance(snapshot[1], CommandCompletedEvent)
        assert snapshot[1].output == "model details"
        assert isinstance(snapshot[2], ModelResponse)
        assert snapshot[2].text == "analysis complete"

    def test_io_snapshot_preserves_messages_before_reset(self):
        agent_session = AgentSession(agent_name="main")
        user_event, request = _user_turn("old question")
        response = ModelResponse(parts=[TextPart(content="old answer")])
        agent_session.add_event(user_event)
        record_messages(agent_session, [request, response])

        compact_history(agent_session, [], True, "compacted-context")

        snapshot = agent_session.build_io_timeline()
        assert len(snapshot) == 3
        assert isinstance(snapshot[0], ModelRequest)
        assert isinstance(snapshot[1], ModelResponse)
        assert snapshot[1].text == "old answer"

    def test_io_snapshot_deduplicates_pending_input_in_same_step(self):
        agent_session = AgentSession(agent_name="main")
        first_event, first_request = _user_turn("first")
        pending_event, pending_request = _user_turn("pending")
        response = ModelResponse(parts=[TextPart(content="done")])
        agent_session.add_event(first_event)
        agent_session.add_event(pending_event)
        record_messages(agent_session, [first_request, pending_request, response])

        snapshot = agent_session.build_io_timeline()

        assert len(snapshot) == 3
        requests = [
            message for message in snapshot if isinstance(message, ModelRequest)
        ]
        assert len(requests) == 2
        assert isinstance(snapshot[2], ModelResponse)

    def test_non_history_events_do_not_change_message_history(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.add_event(UserInputEvent(client_input=_message_input("hello")))
        agent_session.add_event(
            CommandCompletedEvent(
                client_input=_command_input("/info"),
                status="handled",
                output="details",
            )
        )
        agent_session.add_event(ErrorEvent(error="something"))
        assert agent_session.message_history == []

    def test_last_user_messages_update(self):
        agent_session = AgentSession(agent_name="main")
        agent_session.record_user_input(
            _message_input("hello", server_message_id="id1")
        )
        assert agent_session.journal[-1].id == "id1"
        assert agent_session.metadata["last_user_messages"] == ["hello"]

        agent_session.record_user_input(
            _message_input("world", server_message_id="id2")
        )
        assert agent_session.metadata["last_user_messages"] == ["hello", "world"]

        agent_session.record_user_input(
            _message_input("third", server_message_id="id3")
        )
        assert agent_session.metadata["last_user_messages"] == ["world", "third"]

    def test_first_class_task_fields(self):
        agent_session = AgentSession(
            agent_name="main",
            task_name="my_task",
            target="/main/my_task",
            initial_message="start",
        )
        assert agent_session.task_name == "my_task"
        assert agent_session.target == "/main/my_task"
        dumped = agent_session.model_dump()
        assert "last_message" not in dumped
        assert "result" not in dumped
        assert "error" not in dumped

    @pytest.mark.asyncio
    async def test_runtime_excluded_from_serialization(self):
        agent_session = AgentSession(agent_name="main")
        runtime = object()
        agent_session.runtime = runtime

        dumped = agent_session.model_dump(mode="json")
        assert "runtime" not in dumped
        assert agent_session.runtime is runtime

    def test_record_error_event(self):
        agent_session = AgentSession(agent_name="main")

        agent_session.record_error_event("something crashed")
        event = agent_session.journal[-1]
        assert isinstance(event, ErrorEvent)
        assert event.error == "something crashed"

    @pytest.mark.asyncio
    async def test_fork_resets_task_fields(self):
        original = AgentSession(
            agent_name="main",
            task_name="old_task",
        )
        original.add_event(UserInputEvent(client_input=_message_input("hi")))
        forked = await original.fork_at(None)
        assert forked.task_name is None
        assert forked.target is None
        assert forked.initial_message is None

    @pytest.mark.asyncio
    async def test_create_child_session(self, tmp_path):
        store = FileSessionStore()
        store.base_dir = tmp_path / "sessions"
        manager = SessionManager(store)
        manager.register_session_type(AgentSession)
        parent = AgentSession(
            agent_name="main",
            path=["root-id"],
            workspace=str(tmp_path),
            manager=manager,
        )

        child = await parent.create_child_session(
            agent_name="worker",
            agent_source="subagent",
            task_name="sub_task",
            target="/main/sub_task",
            initial_message="do work",
        )

        assert child.owner is parent
        assert child.manager is manager
        assert child.id in parent.children
        assert child.path == ["root-id", child.id]
        assert child.agent_name == "worker"
        assert child.agent_source == "subagent"
        assert child.task_name == "sub_task"
        assert child.target == "/main/sub_task"
        assert child.initial_message == "do work"
        assert child.workspace == str(tmp_path)
        assert child.run_info.llm_context_id is not None
        assert child.run_info.llm_context_id != parent.run_info.llm_context_id

        stored_parent = await store.load_session(parent.path)
        stored_child = await store.load_session(child.path)
        assert stored_parent is not None
        assert stored_parent.children == [child.id]
        assert isinstance(stored_child, AgentSession)
        assert stored_child.task_name == "sub_task"

    @pytest.mark.asyncio
    async def test_create_child_session_workspace_override(self, tmp_path):
        parent = AgentSession(
            agent_name="main",
            path=["root-id"],
            workspace=str(tmp_path),
        )
        custom_ws = tmp_path / "custom"
        child = await parent.create_child_session(
            agent_name="worker",
            workspace=custom_ws,
        )
        assert child.workspace == str(custom_ws.absolute())

    @pytest.mark.asyncio
    async def test_create_runtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        config_file = tmp_path / ".arox" / "config.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("""
model_ref = "test"
[agent.test_worker]
system_prompt = "Hello worker."
""")
        config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
        io_adapter = _StubIOAdapter()

        session = AgentSession(
            agent_name="test_worker",
            path=["worker-session-id"],
        )
        runtime = AgentRuntime(config_loader, io_adapter, session)
        async with runtime:
            assert runtime.uuid == "worker-session-id"
            assert runtime.session is session
            assert not hasattr(runtime, "message_history")
            assert runtime.name == "test_worker"
            assert type(runtime) is AgentRuntime

    @pytest.mark.asyncio
    async def test_ensure_runtime_starts_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        config_file = tmp_path / ".arox" / "config.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("""
model_ref = "test"
[agent.test_worker]
""")
        config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
        io_adapter = _StubIOAdapter()
        session = AgentSession(agent_name="test_worker")

        first, second = await asyncio.gather(
            session.ensure_runtime(config_loader, io_adapter),
            session.ensure_runtime(config_loader, io_adapter),
        )

        assert first is second is session.runtime
        await first.close()

    @pytest.mark.asyncio
    async def test_create_runtime_missing_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        config_file = tmp_path / ".arox" / "config.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("""
model_ref = "test"
""")
        config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
        io_adapter = _StubIOAdapter()

        session = AgentSession(
            agent_name="unconfigured_agent",
        )
        with pytest.raises(
            ValueError, match="Agent config for 'unconfigured_agent' not found"
        ):
            AgentRuntime(config_loader, io_adapter, session)

    @pytest.mark.asyncio
    async def test_agent_config_type_is_not_runtime_dispatch(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        config_file = tmp_path / ".arox" / "config.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("""
model_ref = "test"
[agent.broken_agent]
""")
        config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
        io_adapter = _StubIOAdapter()

        session = AgentSession(
            agent_name="broken_agent",
        )
        runtime = AgentRuntime(config_loader, io_adapter, session)
        async with runtime:
            assert type(runtime) is AgentRuntime


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
        first_input, first_request = _user_turn("hello")
        agent_session.add_event(first_input)
        messages = [
            first_request,
            ModelResponse(parts=[TextPart(content="hi there")]),
        ]
        _record_step(agent_session, messages)
        second_input, _ = _user_turn("follow-up")
        agent_session.add_event(second_input)
        compacted = [
            ModelRequest(parts=[UserPromptPart(content="conversation summary")])
        ]
        compact_history(agent_session, compacted, True, "ctx-compact")

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
        assert [entry.event_type for entry in loaded_agent.journal] == [
            "user_input",
            "model_message",
            "model_message",
            "user_input",
            "compaction",
            "context_reset",
            "model_message",
        ]
        assert "messages" not in context_resets(loaded_agent)[0].model_dump()
        assert loaded_agent.message_history == compacted

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
        agent_s.add_event(UserInputEvent(client_input=_message_input("first")))
        await store.save_session(agent_s)

        agent_s.add_event(UserInputEvent(client_input=_message_input("second")))
        await store.save_session(agent_s)

        loaded = await store.load_session(agent_s.path)
        assert loaded is not None
        assert len(loaded.journal) == 2

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

    @pytest.mark.asyncio
    async def test_manager_does_not_save_empty_root_session(self, store):
        manager = SessionManager(store)
        manager.register_session_type(AgentSession)
        session = AgentSession(agent_name="main", path=["empty"], manager=manager)

        await manager.persist(session)

        assert await store.load_session(session.path) is None
        assert await manager.resolve(session.id) is session

        session.record_user_input(_message_input("hello"))
        await manager.persist(session)

        loaded = await store.load_session(session.path)
        assert isinstance(loaded, AgentSession)
        assert len(loaded.journal) == 1

    @pytest.mark.asyncio
    async def test_manager_tree_api(self, store):
        manager = SessionManager(store)
        manager.register_session_type(AgentSession)
        root = AgentSession(agent_name="main", path=["root"], manager=manager)
        child = await root.create_child_session("worker", task_name="child")
        grandchild = await child.create_child_session("worker", task_name="grandchild")

        loaded = await manager.resolve(root.id)
        assert isinstance(loaded, AgentSession)
        assert loaded is root
        children = await manager.children_of(loaded)
        assert [item.id for item in children] == [child.id]
        assert children[0].owner is loaded
        assert children[0].manager is manager

        walked = await manager.walk(loaded)
        assert [item.id for item in walked] == [root.id, child.id, grandchild.id]
        found = await manager.find(loaded, grandchild.id)
        assert found is walked[-1]

        roots = await manager.list_roots()
        assert roots == [loaded]

        await manager.remove_child(loaded, children[0])
        assert loaded.children == []
        assert await manager.resolve(child.id, loaded) is None
        assert await store.load_session(child.path) is None

    @pytest.mark.asyncio
    async def test_stop_all_stops_active_child_when_root_is_inactive(self, store):
        manager = SessionManager(store)
        manager.register_session_type(AgentSession)
        root = AgentSession(agent_name="main", path=["root"], manager=manager)
        child = await root.create_child_session("worker", task_name="child")
        child_runtime = SimpleNamespace(close=AsyncMock())
        child.runtime = child_runtime
        manager._track(root)
        manager._track(child, root)

        await manager.stop_all()

        child_runtime.close.assert_awaited_once()


def test_message_history_cache_rebuilds_only_the_tail_and_updates_incrementally():
    session = AgentSession(agent_name="main")
    record_messages(session, [ModelRequest.user_text_prompt("old")])
    summary = ModelRequest.user_text_prompt("summary")
    reset_history(session, [summary])
    answer = ModelResponse(parts=[TextPart(content="answer")])
    record_messages(session, [answer])
    restored = AgentSession.model_validate_json(session.model_dump_json())

    class ObservedJournal(list):
        visited = 0

        def __reversed__(self):
            for event in super().__reversed__():
                self.visited += 1
                yield event

    journal = ObservedJournal(restored.journal)
    restored.journal = journal
    assert restored.message_history == [summary, answer]
    assert journal.visited == 3  # Two messages and the reset, never the old history.
    snapshot = restored.message_history
    snapshot.clear()
    assert restored.message_history == [summary, answer]
    assert journal.visited == 3
    cached = restored._message_history
    record_messages(restored, [ModelRequest.user_text_prompt("next")])
    assert restored._message_history is cached
    assert len(restored.message_history) == 3
    restored.add_event(ErrorEvent(error="unrelated"))
    assert len(restored.message_history) == 3
    assert journal.visited == 3
    restored.reset_message_history()
    assert restored.message_history == []
    assert journal.visited == 3
    assert "_message_history" not in restored.model_dump()
    assert "message_history" not in restored.model_dump()


@pytest.mark.asyncio
async def test_fork_discards_parent_history_cache_and_retains_reset_messages():
    session = AgentSession(agent_name="main")
    first, request = _user_turn("first")
    session.add_event(first)
    record_messages(session, [request])
    summary = ModelRequest.user_text_prompt("summary")
    reset_history(session, [summary])
    second, request = _user_turn("second")
    session.add_event(second)
    record_messages(session, [request])
    assert session.message_history == [summary, request]

    forked = await session.fork_at(second.id)
    assert forked.message_history == [summary]
    forked.reset_message_history()
    assert session.message_history == [summary, request]
    assert (await session.fork_at(first.id)).message_history == []
    assert (await session.fork_at(None)).message_history == []


def test_io_timeline_boundary_is_inclusive_and_rejects_missing_ids():
    session = AgentSession(agent_name="main")
    assert session.build_io_timeline() == ()
    with pytest.raises(ValueError, match="not found"):
        session.build_io_timeline(through_id="missing")
    user, request = _user_turn("question")
    session.add_event(user)
    record_messages(session, [request])
    command = session.record_command_completed(
        _command_input("/info"), "handled", output="details"
    )
    response = ModelResponse(parts=[TextPart(content="answer")])
    record_messages(session, [response])
    session.reset_message_history()
    reset_id = session.journal[-1].id
    assert len(session.build_io_timeline(through_id=user.id)) == 1
    before_answer = session.build_io_timeline(through_id=command.id)
    assert len(before_answer) == 2
    assert before_answer[-1] is command
    assert len(session.build_io_timeline(through_id=reset_id)) == 3
    session.record_error_event("later error")
    assert len(session.build_io_timeline(through_id=reset_id)) == 3
    assert len(session.build_io_timeline()) == 4


def test_reset_is_independent_of_its_cause_and_replacement_messages():
    session = AgentSession(agent_name="main")
    record_messages(session, [ModelRequest.user_text_prompt("old")])
    session.record_command_completed(_command_input("/clear"), "handled")
    assert len(session.message_history) == 1
    before_reset = len(session.journal)
    session.reset_message_history()
    assert len(session.journal) == before_reset + 1
    assert session.message_history == []
    assert session.journal[-1].event_type == "context_reset"
    assert "compaction" not in session.journal[-1].model_dump()
    assert "messages" not in session.journal[-1].model_dump()
    assert len(session.build_io_timeline()) == 2  # Old message and the command.
    replacement = ModelRequest.user_text_prompt("new context")
    session.record_model_messages([replacement], run_id="new", context_only=True)
    assert session.message_history == [replacement]
    assert len(session.build_io_timeline()) == 2
    restored = AgentSession.model_validate_json(session.model_dump_json())
    assert restored.message_history == [replacement]
