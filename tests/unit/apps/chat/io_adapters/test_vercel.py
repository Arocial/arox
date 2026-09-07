import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from pydantic_ai import FinalResultEvent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.ui.vercel_ai import VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage

from arox.apps.chat.io_adapters.vercel_ai import (
    CreateSessionRequest,
    VercelStreamIOAdapter,
    VercelStreamServer,
    _log_ws_payload,
    build_state_history,
    dump_ui_messages,
)
from arox.core.agent_runtime import AgentRuntime
from arox.core.app import app_setup
from arox.core.io import AgentIOEndpoint, IOEndpoint, SnapshotEvent
from arox.core.session import (
    MODEL_MESSAGE_ID_KEY,
    AgentSession,
    CommandCompletedEvent,
    ErrorEvent,
    FileSessionStore,
    SessionManager,
    UserInputEvent,
)
from arox.core.types import (
    USER_INPUT_ID_KEY,
    ClientInput,
    CommandPayload,
    MessagePayload,
    TurnStateEvent,
    normalize_client_input,
)
from arox.plugins.compaction import CompactionEvent
from tests.history import compact_history, record_messages, reset_history


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_snapshot", [False, True])
async def test_snapshot_rebuild_stops_at_captured_journal_boundary(empty_snapshot):
    session = AgentSession(agent_name="coder")
    record_messages(session, [ModelRequest.user_text_prompt("committed")])
    boundary = None if empty_snapshot else session.journal[-1].id
    record_messages(session, [ModelResponse(parts=[TextPart(content="later answer")])])
    adapter = VercelStreamIOAdapter()
    endpoint = IOEndpoint()
    websocket = SimpleNamespace(send_json=AsyncMock())
    adapter.adapter_ep_to_runtime[endpoint] = cast(
        AgentRuntime, SimpleNamespace(turn=None)
    )
    adapter._event_contexts[endpoint] = (
        cast(WebSocket, websocket),
        session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
        "test",
        session,
    )
    await adapter.handle_event(endpoint, SnapshotEvent(boundary))
    frame = websocket.send_json.call_args.args[0]
    if empty_snapshot:
        assert frame["history"] == []
    else:
        assert len(frame["history"]) == 1
        assert frame["history"][0]["message"]["parts"][0]["text"] == "committed"


def test_websocket_debug_log_is_structured_and_compact(caplog):
    with caplog.at_level("DEBUG", logger="arox.apps.chat.io_adapters.vercel_ai"):
        _log_ws_payload("IN", "session-1", {"type": "message", "text": "你好"})

    assert (
        "WS IN session_id=session-1 type=message size=30 "
        'payload={"type":"message","text":"你好"}' in caplog.text
    )


def test_websocket_debug_log_truncates_large_payload(caplog):
    with caplog.at_level("DEBUG", logger="arox.apps.chat.io_adapters.vercel_ai"):
        _log_ws_payload("OUT", "session-1", {"type": "data", "text": "x" * 1100})

    assert "WS OUT session_id=session-1 type=data size=1125" in caplog.text
    assert "<truncated 101 chars>" in caplog.text


@pytest.mark.asyncio
async def test_error_event_becomes_error_message():
    adapter = VercelStreamIOAdapter()

    frames = await adapter._to_ui_messages(
        ErrorEvent(error="ValueError: bad response"),
        AgentSession(path=["root"], agent_name="coder"),
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [{"type": "error", "errorText": "ValueError: bad response"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [True, False])
async def test_turn_state_event_becomes_command(busy):
    adapter = VercelStreamIOAdapter()

    frames = await adapter._to_ui_messages(
        TurnStateEvent(busy=busy),
        AgentSession(path=["root"], agent_name="coder"),
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [{"type": "cmd-turn-state", "busy": busy}]


@pytest.mark.asyncio
async def test_compaction_event_becomes_live_marker():
    timestamp = datetime(2026, 9, 7, 12, 30, tzinfo=UTC)
    event = CompactionEvent(
        id="compaction-1",
        timestamp=timestamp,
        trigger="token_threshold",
        llm_context_id="context-2",
    )
    adapter = VercelStreamIOAdapter()

    frames = await adapter._to_ui_messages(
        event,
        AgentSession(path=["root"], agent_name="coder"),
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [
        {
            "type": "compaction",
            "event_id": "compaction-1",
            "trigger": "token_threshold",
            "llm_context_id": "context-2",
            "timestamp": "2026-09-07T12:30:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_final_result_does_not_close_retained_turn_stream():
    adapter = VercelStreamIOAdapter()

    frames = await adapter._to_ui_messages(
        FinalResultEvent(tool_name=None, tool_call_id=None),
        AgentSession(path=["root"], agent_name="coder"),
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == []


@pytest.mark.asyncio
async def test_cancel_uses_runtime_turn_api():
    adapter = VercelStreamIOAdapter()
    runtime = SimpleNamespace(cancel_turn=AsyncMock(return_value=True))
    target = SimpleNamespace(session=SimpleNamespace(runtime=runtime))

    result = await adapter._apply_input(
        target, cast(AgentIOEndpoint, SimpleNamespace()), {"cancel": True}
    )

    assert result is None
    runtime.cancel_turn.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_structured_command_is_forwarded_through_io():
    adapter = VercelStreamIOAdapter()
    command = {"type": "MissingCommand"}
    adapter_ep = SimpleNamespace(send=AsyncMock())

    result = await adapter._apply_input(
        SimpleNamespace(),
        cast(AgentIOEndpoint, adapter_ep),
        {"command": command, "client_message_id": "client-command-1"},
    )

    assert result is None
    event = adapter_ep.send.await_args.args[0]
    assert isinstance(event, ClientInput)
    assert isinstance(event.payload, CommandPayload)
    assert event.payload.command == command
    assert event.client_message_id == "client-command-1"


@pytest.mark.asyncio
async def test_reply_payload_is_forwarded_as_chat_input_event():
    adapter = VercelStreamIOAdapter()
    adapter_ep = SimpleNamespace(send=AsyncMock())
    payload = {
        "reply": {
            "id": "msg-1",
            "role": "user",
            "parts": [{"type": "text", "text": "hello", "state": "done"}],
            "metadata": {
                "custom": {
                    "chatInputEventResult": {"client_message_id": "client-message-1"}
                }
            },
        }
    }

    result = await adapter._apply_input(
        SimpleNamespace(), cast(AgentIOEndpoint, adapter_ep), payload
    )

    assert result is None
    event = adapter_ep.send.await_args.args[0]
    assert isinstance(event, ClientInput)
    assert isinstance(event.payload, MessagePayload)
    assert event.payload.text_content == "hello"
    assert event.client_message_id == "client-message-1"


@pytest.mark.asyncio
async def test_reply_payload_without_chat_input_metadata_is_rejected():
    adapter = VercelStreamIOAdapter()
    adapter_ep = SimpleNamespace(send=AsyncMock())
    payload = {
        "reply": {
            "id": "msg-1",
            "role": "user",
            "parts": [{"type": "text", "text": "hello", "state": "done"}],
        }
    }

    with pytest.raises(ValueError, match="no chatInputEventResult"):
        await adapter._apply_input(
            SimpleNamespace(), cast(AgentIOEndpoint, adapter_ep), payload
        )
    adapter_ep.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_started_message_input_becomes_client_input_frame():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")
    user_input = normalize_client_input(
        ClientInput(
            payload=MessagePayload(content="delegated task"),
            client_message_id="client-message-1",
        )
    )
    assert isinstance(user_input.payload, MessagePayload)
    user_input.payload.status = "started"

    frames = await adapter._to_ui_messages(
        user_input,
        root_session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [
        {
            "type": "cmd-client-input",
            "client_message_id": "client-message-1",
            "server_message_id": user_input.server_message_id,
            "payload": {
                "type": "message",
                "status": "started",
                "message": {
                    "id": user_input.server_message_id,
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "delegated task", "state": "done"}
                    ],
                    "metadata": {
                        "custom": {USER_INPUT_ID_KEY: user_input.server_message_id}
                    },
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_user_message_includes_generated_client_and_server_message_ids():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")
    user_input = normalize_client_input(
        ClientInput(payload=MessagePayload(content="delegated task"))
    )
    assert isinstance(user_input.payload, MessagePayload)
    user_input.payload.status = "started"

    frames = await adapter._to_ui_messages(
        user_input,
        root_session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames[0]["client_message_id"] == user_input.client_message_id
    assert frames[0]["client_message_id"]
    assert frames[0]["server_message_id"] == user_input.server_message_id


@pytest.mark.asyncio
async def test_accepted_command_input_becomes_client_input_frame():
    adapter = VercelStreamIOAdapter()
    client_input = normalize_client_input(
        ClientInput(
            payload=CommandPayload(command={"type": "InfoEvent"}),
            client_message_id="client-command-1",
        )
    )
    assert isinstance(client_input.payload, CommandPayload)
    client_input.payload.status = "accepted"

    frames = await adapter._to_ui_messages(
        client_input,
        AgentSession(path=["root"], agent_name="coder"),
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [
        {
            "type": "cmd-client-input",
            "client_message_id": "client-command-1",
            "server_message_id": client_input.server_message_id,
            "payload": {
                "type": "command",
                "status": "accepted",
                "command": {"type": "InfoEvent"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_command_completion_includes_normalized_input():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")
    client_input = ClientInput(
        payload=CommandPayload(command="/info"),
        client_message_id="client-1",
        server_message_id="server-1",
    )

    frames = await adapter._to_ui_messages(
        CommandCompletedEvent(
            client_input=client_input, status="handled", output="details"
        ),
        root_session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert frames == [
        {
            "type": "cmd-command-completed",
            "input": {
                "payload": {
                    "command": "/info",
                    "status": None,
                    "type": "command",
                },
                "client_message_id": "client-1",
                "server_message_id": "server-1",
            },
            "status": "handled",
            "output": "details",
            "error": None,
        }
    ]


def test_state_timeline_preserves_message_command_order():
    session = AgentSession(path=["root"], agent_name="coder")
    base = datetime.now(UTC)
    message_input = normalize_client_input(
        ClientInput(payload=MessagePayload(content="before"))
    )
    assert message_input.server_message_id is not None
    session.record(
        UserInputEvent(id=message_input.server_message_id, client_input=message_input)
    )
    command_input = normalize_client_input(
        ClientInput(
            payload=CommandPayload(command="/info", status="accepted"),
            client_message_id="client-command-1",
            server_message_id="server-command-1",
        )
    )
    session.record(
        CommandCompletedEvent(
            id="server-command-1",
            client_input=command_input,
            status="handled",
            output="details",
        )
    )
    payload = message_input.payload
    assert isinstance(payload, MessagePayload)
    assert payload.content is not None
    request = ModelRequest(
        parts=[UserPromptPart(content=payload.content)],
        timestamp=base + timedelta(seconds=2),
    )
    response = ModelResponse(
        parts=[TextPart(content="after")],
        timestamp=base,
    )
    record_messages(session, [request, response])

    timeline = build_state_history(session)

    assert [entry["type"] for entry in timeline] == [
        "message",
        "command",
        "message",
    ]
    assert timeline[1]["client_message_id"] == "client-command-1"


def test_state_timeline_includes_compaction_marker_and_stable_message_ids():
    session = AgentSession(path=["root"], agent_name="coder")
    client_input = normalize_client_input(
        ClientInput(payload=MessagePayload(content="question"))
    )
    assert client_input.server_message_id is not None
    session.record(
        UserInputEvent(id=client_input.server_message_id, client_input=client_input)
    )
    response = ModelResponse(parts=[TextPart(content="answer")])
    record_messages(session, [response])
    response_event = next(
        entry for entry in session.journal if entry.event_type == "model_message"
    )
    response_id = response_event.id
    compact_history(session, [], "context-2", trigger="token_threshold")

    timeline = build_state_history(session)

    assert [entry["type"] for entry in timeline] == [
        "message",
        "message",
        "compaction",
    ]
    assert timeline[0]["message"]["id"] == client_input.server_message_id
    assert timeline[1]["message"]["id"] == response_id
    assert timeline[2] == {
        "type": "compaction",
        "event_id": session.journal[-2].id,
        "trigger": "token_threshold",
        "llm_context_id": "context-2",
        "timestamp": session.journal[-2].timestamp.isoformat(),
    }


def test_dump_ui_messages_uses_stored_model_message_id():
    response = ModelResponse(
        parts=[TextPart(content="answer")],
        metadata={MODEL_MESSAGE_ID_KEY: "response-1"},
    )

    history = dump_ui_messages([response])

    assert history[0]["id"] == "response-1"


def test_dump_ui_messages_carries_user_input_id():
    """The history builder must thread user_input_id onto the message metadata."""
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(content="hi\n", metadata={USER_INPUT_ID_KEY: "abc123"})
                ]
            )
        ]
    )

    history = dump_ui_messages([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "abc123"


def test_dump_ui_messages_leaves_untagged_user_message_clean():
    """An untagged user turn must not gain an anchor."""
    request = ModelRequest(
        parts=[UserPromptPart(content=[TextContent(content="hi\n")])]
    )

    history = dump_ui_messages([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata") is None or USER_INPUT_ID_KEY not in msg.get(
        "metadata", {}
    ).get("custom", {})


def test_dump_ui_messages_keeps_anchors_for_identical_text():
    """Identical text in different parts or messages must keep their unique anchors."""
    request1 = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content="same text\n", metadata={USER_INPUT_ID_KEY: "a"}
                    )
                ]
            )
        ]
    )
    request2 = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content="same text\n", metadata={USER_INPUT_ID_KEY: "b"}
                    )
                ]
            )
        ]
    )

    history = dump_ui_messages([request1, request2])
    assert len(history) == 2
    assert (
        history[0].get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "a"
    )
    assert (
        history[1].get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "b"
    )


@pytest.mark.asyncio
async def test_websocket_can_wait_for_active_runtime():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")

    class FakeWebSocket:
        def __init__(self):
            self.accepted = False
            self.sent = []
            self.state_sent = asyncio.Event()
            self.closed = None

        async def accept(self):
            self.accepted = True

        async def receive_json(self):
            await self.state_sent.wait()
            raise WebSocketDisconnect()

        async def send_json(self, payload):
            self.sent.append(payload)
            if payload.get("type") == "state":
                self.state_sent.set()

        async def close(self, code, reason):
            self.closed = (code, reason)

    websocket = FakeWebSocket()
    await adapter.ws_handler(cast(WebSocket, websocket), session, session, "test")

    assert websocket.accepted
    assert websocket.sent == [
        {
            "type": "state",
            "history": [],
            "model": "test",
            "busy": False,
        }
    ]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_websocket_starts_with_runtime_snapshot():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")
    runtime = SimpleNamespace(uuid=session.id)
    record_messages(session, [ModelRequest.user_text_prompt("committed")])
    runtime.agent_ep = AgentIOEndpoint()
    runtime.agent_ep.checkpoint(session.journal[-1].id)
    session.runtime = runtime

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.state_sent = asyncio.Event()

        async def accept(self):
            pass

        async def receive_json(self):
            await self.state_sent.wait()
            raise WebSocketDisconnect()

        async def send_json(self, payload):
            self.sent.append(payload)
            if payload.get("type") == "state":
                self.state_sent.set()

    websocket = FakeWebSocket()
    await adapter.ws_handler(cast(WebSocket, websocket), session, session, "test")

    websocket.sent[0]["history"][0]["message"].pop("id")
    websocket.sent[0]["history"][0]["message"].pop("metadata")
    assert websocket.sent == [
        {
            "type": "state",
            "history": [
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": "committed",
                                "state": "done",
                            }
                        ],
                    },
                }
            ],
            "model": "test",
            "busy": False,
        }
    ]


@pytest.mark.asyncio
async def test_new_websocket_replaces_existing_connection():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")
    runtime = SimpleNamespace(uuid=session.id)
    runtime.agent_ep = AgentIOEndpoint()
    session.runtime = runtime

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.state_sent = asyncio.Event()
            self.closed = None

        async def accept(self):
            pass

        async def receive_json(self):
            await asyncio.Event().wait()

        async def send_json(self, payload):
            self.sent.append(payload)
            if payload.get("type") == "state":
                self.state_sent.set()

        async def close(self, code, reason):
            self.closed = (code, reason)

    first = FakeWebSocket()
    first_task = asyncio.create_task(
        adapter.ws_handler(cast(WebSocket, first), session, session, "test")
    )
    await asyncio.wait_for(first.state_sent.wait(), timeout=1)

    second = FakeWebSocket()
    second_task = asyncio.create_task(
        adapter.ws_handler(cast(WebSocket, second), session, session, "test")
    )
    await asyncio.wait_for(second.state_sent.wait(), timeout=1)

    assert first.closed == (4000, "Replaced by a newer connection.")
    assert first_task.done()
    assert adapter._connections[session.id].task is second_task

    second_task.cancel()
    await asyncio.gather(second_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_websocket_survives_runtime_restart():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")

    first_runtime = SimpleNamespace(uuid=session.id, session=session)
    first_runtime.agent_ep = AgentIOEndpoint()
    session.runtime = first_runtime

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.state_sent = asyncio.Event()

        async def accept(self):
            pass

        async def receive_json(self):
            await asyncio.Event().wait()

        async def send_json(self, payload):
            self.sent.append(payload)
            if payload.get("type") == "state":
                self.state_sent.set()

    websocket = FakeWebSocket()
    websocket_task = asyncio.create_task(
        adapter.ws_handler(cast(WebSocket, websocket), session, session, "test")
    )
    await asyncio.wait_for(websocket.state_sent.wait(), timeout=1)

    await adapter.on_runtime_stop(cast(AgentRuntime, first_runtime))
    assert not websocket_task.done()
    assert adapter._connections[session.id].runtime is None

    websocket.state_sent.clear()
    second_runtime = SimpleNamespace(uuid=session.id, session=session)
    record_messages(session, [ModelRequest.user_text_prompt("after restart")])
    second_runtime.agent_ep = AgentIOEndpoint()
    second_runtime.agent_ep.checkpoint(session.journal[-1].id)
    session.runtime = second_runtime
    await adapter.on_runtime_start(cast(AgentRuntime, second_runtime))
    await asyncio.wait_for(websocket.state_sent.wait(), timeout=1)

    assert adapter._connections[session.id].runtime is second_runtime
    assert (
        websocket.sent[-1]["history"][0]["message"]["parts"][0]["text"]
        == "after restart"
    )

    websocket_task.cancel()
    await asyncio.gather(websocket_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_server_exposes_session_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("""
model_ref = "test"
[app]
main_agent = "coder"
[agent.coder]
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    store = FileSessionStore()
    store.base_dir = tmp_path / "sessions"
    manager = SessionManager(store)
    manager.register_session_type(AgentSession)
    session = AgentSession(
        path=["root-session"],
        agent_name="coder",
        manager=manager,
    )
    reset_history(session, [ModelRequest(parts=[UserPromptPart(content="hello")])])
    await manager.persist(session)

    server = VercelStreamServer(manager, config_loader)
    routes = {getattr(route, "path", None) for route in server.app.routes}
    assert "/api/agents" not in routes
    assert "/api/sessions" in routes
    assert "/api/sessions/{session_id}/start" in routes
    assert "/api/sessions/{root_session_id}/nodes/{target_session_id}/ws" in routes

    views = await server.list_sessions()
    assert len(views) == 1
    assert views[0].id == session.id
    assert not views[0].active

    assert (
        "/api/sessions/{root_session_id}/nodes/{target_session_id}/state" not in routes
    )

    async with manager, server.io_adapter:
        created = await server.create_session(CreateSessionRequest())
        await asyncio.sleep(0)
        assert created.active
        active = await manager.resolve(created.id)
        assert isinstance(active, AgentSession)
        assert active.runtime is not None
        assert not server.io_adapter.adapter_ep_to_runtime

        stopped = await server.stop_session(created.id)
        assert not stopped.active
        inactive = await manager.resolve(created.id)
        assert isinstance(inactive, AgentSession)
        assert not inactive.is_active
        assert await store.load_session(inactive.path) is None

        monkeypatch.setattr(
            server.io_adapter, "suggestions", AsyncMock(return_value={"items": []})
        )
        suggestions = await server.suggestions(inactive.id, inactive.id)
        assert suggestions == {"items": []}
        assert inactive.is_active
        await manager.stop_tree(inactive)
