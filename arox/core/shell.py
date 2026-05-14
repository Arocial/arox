"""Composer-level command shell.

A :class:`ComposerShell` sits alongside the composer's main agent and owns
commands whose scope is the whole composer session (``/rewind`` and future
``/fork``, ``/switch`` etc.) rather than a single agent. It shares the
:class:`IOHost` machinery with :class:`LLMBaseAgent`: its own pair of
:class:`IOEndpoint` and receive loop, driven by the same adapter that
serves the agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from arox.core.io import IOHost
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


class ComposerShell(IOHost):
    """Lightweight host for composer-scope commands and system IO."""

    def __init__(self, composer: Composer):
        super().__init__(composer.io_adapter)
        self.composer = composer
        self.command_manager = CommandManager(self)
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
