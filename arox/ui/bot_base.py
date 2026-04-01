import asyncio
import logging
from abc import ABC, abstractmethod

from anyio import EndOfStream
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

from arox.ui.io import (
    AbstractIOAdapter,
    AdapterIOInterface,
    ChatInputEvent,
)

logger = logging.getLogger(__name__)


class BotIOAdapter(AbstractIOAdapter, ABC):
    def __init__(self, adapter_io: AdapterIOInterface | None = None):
        super().__init__(adapter_io)
        self.message_buffer = []
        self.current_task = None
        self.read_lock = asyncio.Lock()
        self.input_queue: asyncio.Queue | None = None

    @abstractmethod
    async def send_message(self, text: str):
        """Send a message to the user. Must be implemented by subclasses."""

    async def before_handle_output(self) -> bool:
        """Hook called before handling an output event. Can be used to wait for a chat ID.
        Returns True if the event should be processed, False otherwise."""
        return True

    async def run_cancellable(self, task):
        self.current_task = asyncio.create_task(task)
        try:
            return await self.current_task
        except asyncio.CancelledError:
            logger.info("Task cancelled")
            await self.send_message("[Step cancelled]")
        finally:
            self.current_task = None

    async def process_events(self):
        import anyio

        async def process_io(adapter_io):
            try:
                while True:
                    event = await adapter_io.adapter_receive()
                    async with self.read_lock:
                        await self._handle_output(event)
            except EndOfStream:
                pass

        async with anyio.create_task_group() as tg:
            for adapter_io in self.adapter_ios:
                tg.start_soon(process_io, adapter_io)

    async def _handle_output(self, event):
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
        elif isinstance(event, ChatInputEvent):
            if not self.input_queue:
                logger.error("input_queue is not initialized")
                return
            reply = {}
            if event.deferred_tools:
                reply["deferred_tools"] = {}
                for key, tool in event.deferred_tools.items():
                    await self.send_message(f"❓ {tool.question}")
                    line = await self.input_queue.get()
                    reply["deferred_tools"][key] = line
            if event.exception_input.exception is not None:
                await self.send_message(
                    f"⚠️ An error occurred: {event.exception_input.exception}\nDo you want to continue? (y/n)"
                )
                line = await self.input_queue.get()
                reply["exception_input"] = {"to_continue": line.strip().lower() == "y"}
            if event.normal_input.request:
                line = await self.input_queue.get()
                reply["normal_input"] = {"user_input": line}
            event.set_reply(reply)
