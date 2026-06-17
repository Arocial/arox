from dataclasses import dataclass

from arox.core.io import IOEndpoint


@dataclass
class UserInput:
    """A unit of user input passed to :meth:`LLMBaseAgent.step`.

    ``client_message_id`` is an opaque id assigned by a client to the message that
    produced this input; it is echoed back in :class:`ServerIdMapping` so the client
    can map its own messages to backend session-event ids.
    """

    user_input: str | None = None
    client_message_id: str | None = None


@dataclass
class ServerIdMapping:
    """Maps a UI-assigned ``message_id`` to the ``event_id`` of the recorded
    user-input session event, so the UI can resolve stable backend event ids
    (used for forking) without relying on positional ordering."""

    event_id: str | None = None
    client_id: str | None = None


@dataclass
class AgentInfoUpdate:
    """Carries updated agent info to broadcast to clients."""

    agent_id: str


@dataclass
class AgentDeps:
    agent_io: IOEndpoint
