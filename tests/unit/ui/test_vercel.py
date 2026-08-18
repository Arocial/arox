import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.ui.vercel_ai import VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage

from arox.core.agent_runtime import AgentRuntime
from arox.core.app import app_setup
from arox.core.io import AgentIOEndpoint
from arox.core.session import AgentSession, ErrorEvent, FileSessionStore, SessionManager
from arox.core.types import USER_INPUT_ID_KEY, UserInput, UserMessageEvent
from arox.ui.vercel_ai import (
    CreateSessionRequest,
    VercelStreamIOAdapter,
    VercelStreamServer,
    build_state_history,
)


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
async def test_user_message_event_becomes_command_with_complete_ui_message():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")
    user_input = UserInput(
        input_content="delegated task", client_message_id="client-message-1"
    )

    frames = await adapter._to_ui_messages(
        UserMessageEvent(user_input),
        root_session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    message_id = frames[0]["message"].pop("id")
    assert isinstance(message_id, str)
    assert message_id != user_input.server_message_id
    assert frames == [
        {
            "type": "cmd-user-message",
            "client_message_id": "client-message-1",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "delegated task", "state": "done"}],
                "metadata": {
                    "custom": {USER_INPUT_ID_KEY: user_input.server_message_id}
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_user_message_omits_client_message_id():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")

    frames = await adapter._to_ui_messages(
        UserMessageEvent(UserInput(input_content="delegated task")),
        root_session,
        VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
    )

    assert "client_message_id" not in frames[0]


def test_build_state_history_carries_user_input_id():
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

    history = build_state_history([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "abc123"


def test_build_state_history_untagged_user_message_stays_clean():
    """An untagged user turn must not gain an anchor."""
    request = ModelRequest(
        parts=[UserPromptPart(content=[TextContent(content="hi\n")])]
    )

    history = build_state_history([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata") is None or USER_INPUT_ID_KEY not in msg.get(
        "metadata", {}
    ).get("custom", {})


def test_build_state_history_identical_text_different_anchors():
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

    history = build_state_history([request1, request2])
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
    assert websocket.sent == [{"type": "state", "history": [], "model": "test"}]
    assert websocket.closed is None


@pytest.mark.asyncio
async def test_websocket_starts_with_runtime_snapshot():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")
    runtime = SimpleNamespace(uuid=session.id)
    runtime.agent_ep = AgentIOEndpoint()
    runtime.agent_ep.snapshot(
        (ModelRequest(parts=[UserPromptPart(content="committed")]),)
    )
    session.runner = SimpleNamespace(runtime=runtime)

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

    websocket.sent[0]["history"][0].pop("id")
    assert websocket.sent == [
        {
            "type": "state",
            "history": [
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": "committed", "state": "done"}],
                }
            ],
            "model": "test",
        }
    ]


@pytest.mark.asyncio
async def test_new_websocket_replaces_existing_connection():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")
    runtime = SimpleNamespace(uuid=session.id)
    runtime.agent_ep = AgentIOEndpoint()
    runtime.agent_ep.snapshot(())
    session.runner = SimpleNamespace(runtime=runtime)

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
    first_runtime.agent_ep.snapshot(())
    session.runner = SimpleNamespace(runtime=first_runtime)

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
    second_runtime.agent_ep = AgentIOEndpoint()
    second_runtime.agent_ep.snapshot(
        (ModelRequest(parts=[UserPromptPart(content="after restart")]),)
    )
    session.runner = SimpleNamespace(runtime=second_runtime)
    await adapter.on_runtime_start(cast(AgentRuntime, second_runtime))
    await asyncio.wait_for(websocket.state_sent.wait(), timeout=1)

    assert adapter._connections[session.id].runtime is second_runtime
    assert websocket.sent[-1]["history"][0]["parts"][0]["text"] == "after restart"

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
    session.message_history.messages = [
        ModelRequest(parts=[UserPromptPart(content="hello")])
    ]
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
        assert active.runner.task is not None
        assert not server.io_adapter.adapter_ep_to_runtime

        stopped = await server.stop_session(created.id)
        assert not stopped.active
        inactive = await manager.resolve(created.id)
        assert isinstance(inactive, AgentSession)
        assert not inactive.is_active
