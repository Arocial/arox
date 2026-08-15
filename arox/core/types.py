import uuid
from dataclasses import dataclass, field
from typing import Sequence

from pydantic_ai import UserContent
from pydantic_ai.messages import TextContent

USER_INPUT_ID_KEY = "user_input_id"


@dataclass
class UserInput:
    """A unit of user input passed to :meth:`AgentRuntime.run`.

    ``client_message_id`` is an opaque id assigned by a client to the message that
    produced this input; it is echoed back in :class:`ServerIdMapping` so the client
    can map its own messages to backend session-event ids.
    """

    input_content: Sequence[UserContent] | str | None = None
    client_message_id: str | None = None
    server_message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if isinstance(self.input_content, str):
            self.input_content = [
                TextContent(
                    content=self.input_content,
                    metadata={USER_INPUT_ID_KEY: self.server_message_id},
                )
            ]

    @property
    def text_content(self) -> str | None:
        if self.input_content is None:
            return None
        for c in self.input_content:
            if isinstance(c, TextContent):
                return c.content


@dataclass
class ServerIdMapping:
    """Maps a UI-assigned ``client_message_id`` to the ``server_message_id`` of the recorded
    user-input session event, so the UI can resolve stable backend event ids
    (used for forking) without relying on positional ordering."""

    server_message_id: str | None = None
    client_message_id: str | None = None


@dataclass
class UserMessageEvent:
    """Requests that an adapter render a runtime-originated user message."""

    user_input: UserInput


@dataclass
class SessionTreeUpdate:
    """Requests broadcasting the latest tree for a root session."""

    session_id: str
