import asyncio
import logging
import os

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
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from arox.ui.io import (
    AbstractIOAdapter,
    AdapterIOInterface,
    ChatInputEvent,
)

logger = logging.getLogger(__name__)


class TelegramIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface):
        super().__init__(adapter_io)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if self.allowed_chat_id:
            self.allowed_chat_id = int(self.allowed_chat_id)

        self.app = None
        self.current_chat_id = self.allowed_chat_id
        self.message_buffer = []
        self.current_task = None
        self.read_lock = asyncio.Lock()
        self.input_queue = asyncio.Queue()
        self.chat_id_event = asyncio.Event()
        if self.current_chat_id:
            self.chat_id_event.set()

    def setup(self, agent):
        pass

    async def start(self):
        if not self.token:
            logger.error(
                "TELEGRAM_BOT_TOKEN is not set. Telegram adapter will not start."
            )
            return

        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        asyncio.create_task(self.process_events())

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if self.allowed_chat_id and chat_id != self.allowed_chat_id:
            await update.message.reply_text("Unauthorized.")
            return
        self.current_chat_id = chat_id
        self.chat_id_event.set()
        await update.message.reply_text("Agent is ready. Send a message to start.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if self.allowed_chat_id and chat_id != self.allowed_chat_id:
            return

        self.current_chat_id = chat_id
        self.chat_id_event.set()
        text = update.message.text
        await self.input_queue.put(text)

    async def run_cancellable(self, task):
        self.current_task = asyncio.create_task(task)
        try:
            return await self.current_task
        except asyncio.CancelledError:
            logger.info("Task cancelled")
            if self.current_chat_id and self.app:
                await self.app.bot.send_message(
                    chat_id=self.current_chat_id, text="[Step cancelled]"
                )
        finally:
            self.current_task = None

    async def process_events(self):
        try:
            while True:
                async with self.read_lock:
                    event = await self.adapter_io.adapter_receive()
                await self._handle_output(event)
        except EndOfStream:
            pass

    async def _handle_output(self, event):
        if not self.app:
            return

        await self.chat_id_event.wait()

        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                self.message_buffer.append(part.content)
            elif isinstance(part, ThinkingPart):
                self.message_buffer.append(f"🤔 Thinking...\n{part.content}")
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, (TextPartDelta, ThinkingPartDelta)):
                self.message_buffer.append(delta.content_delta)
        elif isinstance(event, PartEndEvent):
            if self.message_buffer:
                text = "".join(self.message_buffer)
                if text.strip():
                    for i in range(0, len(text), 4000):
                        await self.app.bot.send_message(
                            chat_id=self.current_chat_id, text=text[i : i + 4000]
                        )
                self.message_buffer = []
        elif isinstance(event, FunctionToolResultEvent):
            result_text = f"🔧 Tool result: {str(event.result.content)[:500]}"
            await self.app.bot.send_message(
                chat_id=self.current_chat_id, text=result_text
            )
        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            call_text = f"🛠 Tool call: {part.tool_name}\nArgs: {str(part.args)[:500]}"
            await self.app.bot.send_message(
                chat_id=self.current_chat_id, text=call_text
            )
        elif isinstance(event, ChatInputEvent):
            reply = {}
            if event.deferred_tools:
                reply["deferred_tools"] = {}
                for key, tool in event.deferred_tools.items():
                    await self.app.bot.send_message(
                        chat_id=self.current_chat_id, text=f"❓ {tool.question}"
                    )
                    line = await self.input_queue.get()
                    reply["deferred_tools"][key] = line
            if event.exception_input.exception is not None:
                await self.app.bot.send_message(
                    chat_id=self.current_chat_id,
                    text=f"⚠️ An error occurred: {event.exception_input.exception}\nDo you want to continue? (y/n)",
                )
                line = await self.input_queue.get()
                reply["exception_input"] = {"to_continue": line.strip().lower() == "y"}
            if event.normal_input.request:
                line = await self.input_queue.get()
                reply["normal_input"] = {"user_input": line}
            event.set_reply(reply)
