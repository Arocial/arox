import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

from arox.core.app import app_setup
from arox.core.io import IOEndpoint
from arox.core.session import AgentSession, FileSessionStore, SessionManager
from arox.core.types import USER_INPUT_ID_KEY, UserInput, UserMessageEvent
from arox.ui.vercel_ai import (
    CreateSessionRequest,
    VercelStreamIOAdapter,
    VercelStreamServer,
    build_state_history,
)


@pytest.mark.asyncio
async def test_user_message_event_becomes_command_with_complete_ui_message():
    adapter = VercelStreamIOAdapter()
    root_session = AgentSession(path=["root"], agent_name="coder")
    user_input = UserInput(input_content="delegated task")

    frames = await adapter._to_ui_messages(
        cast(IOEndpoint, object()),
        UserMessageEvent(user_input),
        root_session,
    )

    assert frames == [
        {
            "type": "cmd-user-message",
            "message": {
                "id": user_input.server_message_id,
                "role": "user",
                "parts": [{"type": "text", "text": "delegated task", "state": "done"}],
                "metadata": {
                    "custom": {USER_INPUT_ID_KEY: user_input.server_message_id}
                },
            },
        }
    ]


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
async def test_adapter_routes_runtime_events_through_session_queue():
    adapter = VercelStreamIOAdapter()
    session_id = "session-id"
    first_ep = SimpleNamespace(host=SimpleNamespace(uuid=session_id))
    second_ep = SimpleNamespace(host=SimpleNamespace(uuid=session_id))

    await adapter.handle_event(cast(IOEndpoint, first_ep), "first")
    await adapter.handle_event(cast(IOEndpoint, second_ep), "second")

    queue = adapter.event_queues[session_id]
    assert await queue.get() == (first_ep, "first")
    assert await queue.get() == (second_ep, "second")


@pytest.mark.asyncio
async def test_websocket_stays_available_without_runtime():
    adapter = VercelStreamIOAdapter()
    session = AgentSession(path=["root"], agent_name="coder")

    class FakeWebSocket:
        def __init__(self):
            self.accepted = False
            self.sent = []
            self.payloads = [{"cancel": True}]

        async def accept(self):
            self.accepted = True

        async def receive_json(self):
            if self.payloads:
                return self.payloads.pop(0)
            raise WebSocketDisconnect()

        async def send_json(self, payload):
            self.sent.append(payload)

    websocket = FakeWebSocket()
    await adapter.ws_handler(cast(WebSocket, websocket), session, session)

    assert websocket.accepted
    assert websocket.sent == [{"type": "ack", "status": "unavailable"}]


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

    state = await server.state(session.id, session.id)
    assert state["history"][0]["role"] == "user"
    assert state["model"] == "test"

    async with manager, server.io_adapter:
        created = await server.create_session(CreateSessionRequest())
        await asyncio.sleep(0)
        assert created.active
        active = await manager.resolve(created.id)
        assert isinstance(active, AgentSession)
        assert active.runner.task is not None

        stopped = await server.stop_session(created.id)
        assert not stopped.active
        inactive = await manager.resolve(created.id)
        assert isinstance(inactive, AgentSession)
        assert not inactive.is_active
