from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arox.core.composer import Composer, ForkEvent
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
            self.command_manager.register(ForkEvent, self.handle_fork)

        @property
        def agent_session(self):
            return self.session

        async def handle_fork(self, event: ForkEvent) -> str:
            return await Composer.handle_fork(self, event)  # type: ignore

    return MockComposer()


def test_fork_event_parsing():
    assert ForkEvent.from_slash("fork", "").n == 1
    assert ForkEvent.from_slash("fork", "3").n == 3
    e = ForkEvent.from_slash("fork", "@5")
    assert e.n is None and e.event_index == 5
    # Bad input falls back to default.
    assert ForkEvent.from_slash("fork", "abc").n == 1


def test_shell_registers_fork():
    ag = AgentSession(agent_name="main")
    composer = _make_composer(ag)
    assert "fork" in composer.command_manager.command_map


@pytest.mark.asyncio
async def test_handle_fork_relative_success():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    ag.add_event("user_input", {"text": "again"})
    composer = _make_composer(ag)

    msg = await composer.handle_fork(ForkEvent(n=1))
    assert "forked:main:2" in msg

    msg = await composer.handle_fork(ForkEvent(n=2))
    assert "forked:main:0" in msg


@pytest.mark.asyncio
async def test_handle_fork_absolute_anchor_check():
    ag = AgentSession(agent_name="main")
    ag.add_event("user_input", {"text": "hi"})
    ag.add_event("agent_step", {})
    composer = _make_composer(ag)

    # @0 is a user-turn anchor → succeeds.
    msg = await composer.handle_fork(ForkEvent(n=None, event_index=0))
    assert "forked:main:0" in msg

    # @1 is an agent_step, not an anchor → rejected.
    msg = await composer.handle_fork(ForkEvent(n=None, event_index=1))
    assert "not a user-turn anchor" in msg


@pytest.mark.asyncio
async def test_handle_fork_not_enough_history():
    ag = AgentSession(agent_name="main")
    composer = _make_composer(ag)

    msg = await composer.handle_fork(ForkEvent(n=1))
    assert "not enough history" in msg
