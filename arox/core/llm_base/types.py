import uuid
from dataclasses import dataclass, field

from pydantic_ai import UserContent
from pydantic_ai.messages import TextContent

from arox.core.session import USER_INPUT_ID_KEY


@dataclass
class UserInput:
    """A unit of user input passed to :meth:`LLMBaseAgent.step`.

    ``client_message_id`` is an opaque id assigned by a client to the message that
    produced this input; it is echoed back in :class:`ServerIdMapping` so the client
    can map its own messages to backend session-event ids.
    """

    user_input: str | list[UserContent] | None = None
    client_message_id: str | None = None
    server_message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_user_content(self) -> list[UserContent] | None:
        if self.user_input is None:
            return None
        if isinstance(self.user_input, str):
            return [
                TextContent(
                    content=self.user_input + "\n",
                    metadata={USER_INPUT_ID_KEY: self.server_message_id},
                )
            ]
        return self.user_input


@dataclass
class ServerIdMapping:
    """Maps a UI-assigned ``client_message_id`` to the ``server_message_id`` of the recorded
    user-input session event, so the UI can resolve stable backend event ids
    (used for forking) without relying on positional ordering."""

    server_message_id: str | None = None
    client_message_id: str | None = None


@dataclass
class AgentInfoUpdate:
    """Carries updated agent info to broadcast to clients."""

    agent_uuid: str
