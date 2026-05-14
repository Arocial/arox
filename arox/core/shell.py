"""Composer-level command shell.

A :class:`ComposerShell` sits alongside the composer's main agent and owns
commands whose scope is the whole composer session (``/rewind`` and future
``/fork``, ``/switch`` etc.) rather than a single agent. It has its own
``CommandManager``, its own pair of :class:`IOEndpoint` for system messages
back to the UI, and a placeholder run loop kept symmetrical with the agent's.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from anyio import ClosedResourceError, EndOfStream

from arox.core.io import create_io_channel
from arox.core.plugin import CommandEvent, CommandManager

if TYPE_CHECKING:
    from arox.core.composer import Composer

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RewindEvent(CommandEvent):
    slashes: ClassVar[tuple[str, ...]] = ("rewind",)
    description: ClassVar[str] = (
        "Rewind to a user turn - /rewind [N] (relative) or /rewind @<index> (absolute)"
    )

    # Exactly one of these is meaningful per event.
    n: int | None = 1
    event_index: int | None = None

    @classmethod
    def from_slash(cls, name, arg):
        raw = (arg or "").strip()
        if not raw:
            return cls(n=1)
        if raw.startswith("@"):
            try:
                return cls(n=None, event_index=int(raw[1:]))
            except ValueError:
                return cls(n=1)
        try:
            return cls(n=max(int(raw), 1))
        except ValueError:
            return cls(n=1)


class ComposerShell:
    """Lightweight host for composer-scope commands and system IO."""

    def __init__(self, composer: Composer):
        self.composer = composer
        self.agent_io, self.adapter_io = create_io_channel()
        self.command_manager = CommandManager(self)
        self._stack = contextlib.AsyncExitStack()
        self._register_builtin_commands()

    # CommandManager records the parsed command as a session event via
    # ``host.agent_session.add_event``; we route those onto the composer
    # session so shell commands show up in the global event log.
    @property
    def agent_session(self):
        return self.composer.session

    @property
    def name(self) -> str:
        return f"{self.composer.name}/shell"

    def _register_builtin_commands(self):
        self.command_manager.register(RewindEvent, self.handle_rewind)

    async def handle_rewind(self, event: RewindEvent) -> str:
        main_agent = self.composer.main_agent
        agent_session = main_agent.agent_session
        if event.event_index is not None:
            target = event.event_index
            anchors = set(agent_session.user_turn_anchors())
            if target not in anchors:
                return f"Cannot rewind to @{target}: not a user-turn anchor."
        else:
            n = event.n or 1
            resolved = agent_session.resolve_user_turn(n)
            if resolved is None:
                return f"Cannot rewind {n} user turn(s): not enough history."
            target = resolved
        new_id = await self.composer.fork_session(main_agent.name, target)
        return (
            f"Forked at event @{target}. New branch session id: {new_id}\n"
            f"Resume with: --resume {new_id}"
        )

    async def __aenter__(self):
        tg = asyncio.TaskGroup()
        await self._stack.enter_async_context(tg)
        tg.create_task(self.composer.io_adapter._process_io(self.adapter_io))
        await self._stack.enter_async_context(self.agent_io)
        await self._stack.enter_async_context(self.adapter_io)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()

    async def run(self):
        """Passive loop: drain shell-side events until cancellation.

        At present nothing routes events into the shell's receive side;
        the loop exists so the shell has a symmetric lifecycle with the
        main agent and can be cancelled cleanly on shutdown.
        """
        try:
            while True:
                event = await self.agent_io.receive()
                logger.debug("ComposerShell received unhandled event: %r", event)
        except (EndOfStream, ClosedResourceError, asyncio.CancelledError):
            pass
