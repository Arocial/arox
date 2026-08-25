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

from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)
from arox.core.session import ErrorEvent
from arox.core.types import ClientInput, MessagePayload

logger = logging.getLogger(__name__)


class BotIOAdapter(AbstractIOAdapter, ABC):
    def __init__(self):
        super().__init__()
        self.message_buffer = []
        self.read_lock = asyncio.Lock()

    @abstractmethod
    async def send_message(self, text: str):
        """Send a message to the user. Must be implemented by subclasses."""

    async def before_handle_output(self) -> bool:
        """Hook called before handling an output event. Can be used to wait for a chat ID.
        Returns True if the event should be processed, False otherwise."""
        return True

    @override
    async def handle_event(self, adapter_ep: IOEndpoint, event):
        async with self.read_lock:
            await self._handle_output(adapter_ep, event)

    async def _handle_output(self, adapter_ep: IOEndpoint, event):
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
        elif isinstance(event, ErrorEvent):
            await self.send_message(f"⚠️ An error occurred: {event.error}")

    async def send_user_input(self, text: str) -> None:
        for adapter_ep, runtime in list(self.adapter_ep_to_runtime.items()):
            if runtime.session.owner is None:
                await adapter_ep.send(ClientInput(payload=MessagePayload(content=text)))
