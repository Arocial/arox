from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.session import AgentSession, ComposerSession
from arox.core.shell import ComposerShell, RewindEvent


def _make_composer(main_agent_session: AgentSession):
    session = ComposerSession(
        id="s1",
        composer_name="test",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        agent_sessions={main_agent_session.agent_name: main_agent_session},
    )
    main_agent = SimpleNamespace(
        name=main_agent_session.agent_name,
        agent_session=main_agent_session,
        plugins=[],
    )

    async def fork_session(agent_name: str, event_index: int) -> str:
        return f"forked:{agent_name}:{event_index}"

    return SimpleNamespace(
        name="test",
        session=session,
        main_agent=main_agent,
        fork_session=fork_session,
        io_adapter=SimpleNamespace(),
    )


def test_rewind_event_parsing():
    assert RewindEvent.from_slash("rewind", "").n == 1
    assert RewindEvent.from_slash("rewind", "3").n == 3
    e = RewindEvent.from_slash("rewind", "@5")
    assert e.n is None and e.event_index == 5
    # Bad input falls back to default.
    assert RewindEvent.from_slash("rewind", "abc").n == 1


def test_shell_registers_rewind():
    ag = AgentSession(agent_name="main")
    composer = _make_composer(ag)
    shell = ComposerShell(composer)  # type: ignore[arg-type]
    assert "rewind" in shell.command_manager.command_map


@pytest.mark.asyncio
async def test_handle_rewind_relative_success():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    ag.add_event("user_input", {"text": "again"})
    composer = _make_composer(ag)
    shell = ComposerShell(composer)  # type: ignore[arg-type]

    msg = await shell.handle_rewind(RewindEvent(n=1))
    assert "forked:main:2" in msg

    msg = await shell.handle_rewind(RewindEvent(n=2))
    assert "forked:main:0" in msg


@pytest.mark.asyncio
async def test_handle_rewind_absolute_anchor_check():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    composer = _make_composer(ag)
    shell = ComposerShell(composer)  # type: ignore[arg-type]

    # @0 is a user-turn anchor → succeeds.
    msg = await shell.handle_rewind(RewindEvent(n=None, event_index=0))
    assert "forked:main:0" in msg

    # @1 is an agent_step, not an anchor → rejected.
    msg = await shell.handle_rewind(RewindEvent(n=None, event_index=1))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_rewind_not_enough_history():
    ag = AgentSession(agent_name="main")
    composer = _make_composer(ag)
    shell = ComposerShell(composer)  # type: ignore[arg-type]

    msg = await shell.handle_rewind(RewindEvent(n=1))
    assert "not enough history" in msg
