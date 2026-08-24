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
    produced this input. When the client does not provide one, Arox generates it.
    Adapters echo it alongside the stable ``server_message_id`` carried in the
    rendered message metadata.
    """

    input_content: Sequence[UserContent] | str | None = None
    client_message_id: str | None = None
    server_message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.client_message_id:
            self.client_message_id = str(uuid.uuid4())
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
class UserMessageEvent:
    """Requests that an adapter render a runtime-originated user message."""

    user_input: UserInput


@dataclass(frozen=True)
class TurnStateEvent:
    """Reports whether the runtime's retained turn is currently executing."""

    busy: bool


@dataclass
class SessionTreeUpdate:
    """Requests broadcasting the latest tree for a root session."""

    session_id: str
