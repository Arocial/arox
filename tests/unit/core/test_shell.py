from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.composer import Composer, RewindEvent
from arox.core.session import AgentSession, ComposerSession


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

    # We mock Composer to avoid full initialization
    class MockComposer:
        def __init__(self):
            self.name = "test"
            self.session = session
            self.main_agent = main_agent
            self.fork_session = fork_session
            self.io_adapter = SimpleNamespace()

            from arox.core.plugin import CommandManager

            self.command_manager = CommandManager(self)
            self.command_manager.register(RewindEvent, self.handle_rewind)

        @property
        def agent_session(self):
            return self.session

        async def handle_rewind(self, event: RewindEvent) -> str:
            return await Composer.handle_rewind(self, event)  # type: ignore

    return MockComposer()


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
    assert "rewind" in composer.command_manager.command_map


@pytest.mark.asyncio
async def test_handle_rewind_relative_success():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    ag.add_event("user_input", {"text": "again"})
    composer = _make_composer(ag)

    msg = await composer.handle_rewind(RewindEvent(n=1))
    assert "forked:main:2" in msg

    msg = await composer.handle_rewind(RewindEvent(n=2))
    assert "forked:main:0" in msg


@pytest.mark.asyncio
async def test_handle_rewind_absolute_anchor_check():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    composer = _make_composer(ag)

    # @0 is a user-turn anchor → succeeds.
    msg = await composer.handle_rewind(RewindEvent(n=None, event_index=0))
    assert "forked:main:0" in msg

    # @1 is an agent_step, not an anchor → rejected.
    msg = await composer.handle_rewind(RewindEvent(n=None, event_index=1))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_rewind_not_enough_history():
    ag = AgentSession(agent_name="main")
    composer = _make_composer(ag)

    msg = await composer.handle_rewind(RewindEvent(n=1))
    assert "not enough history" in msg
