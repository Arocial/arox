import uuid
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

from pydantic_ai import UserContent
from pydantic_ai.messages import TextContent

USER_INPUT_ID_KEY = "user_input_id"


@dataclass
class MessagePayload:
    content: Sequence[UserContent] | str | None = None
    status: Literal["started"] | None = None
    type: Literal["message"] = "message"

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            self.content = [TextContent(content=self.content)]

    @property
    def text_content(self) -> str | None:
        if self.content is None:
            return None
        for item in self.content:
            if isinstance(item, TextContent):
                return item.content
        return None


@dataclass
class CommandPayload:
    command: str | dict
    status: Literal["accepted"] | None = None
    type: Literal["command"] = "command"


ClientInputPayload: TypeAlias = MessagePayload | CommandPayload


@dataclass(kw_only=True)
class ClientInput:
    """One client submission, before or after runtime normalization."""

    payload: ClientInputPayload
    client_message_id: str | None = None
    server_message_id: str | None = None


def normalize_client_input(client_input: ClientInput) -> ClientInput:
    """Fill stable ids and attach the server id to message content."""
    if not client_input.client_message_id:
        client_input.client_message_id = str(uuid.uuid4())
    if not client_input.server_message_id:
        client_input.server_message_id = str(uuid.uuid4())

    payload = client_input.payload
    if isinstance(payload, MessagePayload):
        text = payload.text_content
        if (
            text
            and text.startswith("/")
            and payload.content is not None
            and len(payload.content) == 1
            and isinstance(payload.content[0], TextContent)
        ):
            client_input.payload = CommandPayload(command=text)
        elif payload.content is not None:
            for item in payload.content:
                if isinstance(item, TextContent):
                    metadata = dict(item.metadata or {})
                    metadata[USER_INPUT_ID_KEY] = client_input.server_message_id
                    item.metadata = metadata
    return client_input


@dataclass(frozen=True)
class TurnStateEvent:
    """Reports whether the runtime's retained turn is currently executing."""

    busy: bool


@dataclass
class SessionTreeUpdate:
    """Requests broadcasting the latest tree for a root session."""

    session_id: str
