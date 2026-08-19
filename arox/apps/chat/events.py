from dataclasses import dataclass

from arox.core.io import ReplyEvent, RequestEvent
from arox.core.types import UserInput


class StepDoneEvent:
    pass


@dataclass
class ChatInputRequest(RequestEvent):
    request_normal_input: bool = True


@dataclass
class ChatInputReply(UserInput, ReplyEvent):
    def is_abort(self, request: ChatInputRequest) -> bool:
        return bool(request.request_normal_input and self.input_content is None)
