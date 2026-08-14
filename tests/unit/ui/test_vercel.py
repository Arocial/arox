import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

from arox.core.app import app_setup
from arox.core.session import AgentSession, FileSessionStore, SessionManager
from arox.core.types import USER_INPUT_ID_KEY
from arox.ui.vercel_ai import (
    CreateSessionRequest,
    VercelStreamServer,
    build_state_history,
)


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
        assert active.runner.serve_task is not None

        stopped = await server.stop_session(created.id)
        assert not stopped.active
        inactive = await manager.resolve(created.id)
        assert isinstance(inactive, AgentSession)
        assert not inactive.is_active
