import asyncio
import logging
from abc import ABC, abstractmethod
from typing import override

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from arox.core.chat import ChatInputReply, ChatInputRequest
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)

logger = logging.getLogger(__name__)


class BotIOAdapter(AbstractIOAdapter, ABC):
    def __init__(self):
        super().__init__()
        self.message_buffer = []
        self.read_lock = asyncio.Lock()
        self.input_queue: asyncio.Queue | None = None

    @abstractmethod
    async def send_message(self, text: str):
        """Send a message to the user. Must be implemented by subclasses."""

    async def before_handle_output(self) -> bool:
        """Hook called before handling an output event. Can be used to wait for a chat ID.
        Returns True if the event should be processed, False otherwise."""
        return True

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event):
        async with self.read_lock:
            await self._handle_output(adapter_io, event)

    async def _handle_output(self, adapter_io: IOEndpoint, event):
        if not await self.before_handle_output():
            return

        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                self.message_buffer.append(part.content)
            elif isinstance(part, ThinkingPart):
                self.message_buffer.append(f"🤔 Thinking...\n{part.content}")
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, (TextPartDelta, ThinkingPartDelta)):
                if delta.content_delta:
                    self.message_buffer.append(delta.content_delta)
        elif isinstance(event, PartEndEvent):
            if self.message_buffer:
                text = "".join(self.message_buffer)
                if text.strip():
                    for i in range(0, len(text), 4000):
                        await self.send_message(text[i : i + 4000])
                self.message_buffer = []
        elif isinstance(event, FunctionToolResultEvent):
            result_text = f"🔧 Tool result: {str(event.result.content)[:500]}"
            await self.send_message(result_text)
        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            call_text = f"🛠 Tool call: {part.tool_name}\nArgs: {str(part.args)[:500]}"
            await self.send_message(call_text)
        elif isinstance(event, ChatInputRequest):
            if not self.input_queue:
                logger.error("input_queue is not initialized")
                return
            user_input: str | None = None
            if event.pending_exception is not None:
                await self.send_message(
                    f"⚠️ An error occurred: {event.pending_exception}"
                )
            if event.request_normal_input:
                while True:
                    line = await self.input_queue.get()
                    user_input = line
                    break
            await adapter_io.send(
                ChatInputReply(
                    req_id=event.req_id,
                    input_content=user_input,
                )
            )
