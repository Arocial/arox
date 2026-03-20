import asyncio
import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from arox.ui.bot_base import BotIOAdapter
from arox.ui.io import AdapterIOInterface

logger = logging.getLogger(__name__)


class TelegramIOAdapter(BotIOAdapter):
    _shared_app = None
    _app_lock = asyncio.Lock()
    _adapters = []
    _shared_input_queue = asyncio.Queue()

    def __init__(self, adapter_io: AdapterIOInterface | None = None):
        super().__init__(adapter_io)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if self.allowed_chat_id:
            self.allowed_chat_id = int(self.allowed_chat_id)

        self.current_chat_id = self.allowed_chat_id
        self.input_queue = TelegramIOAdapter._shared_input_queue
        self.chat_id_event = asyncio.Event()
        if self.current_chat_id:
            self.chat_id_event.set()

        TelegramIOAdapter._adapters.append(self)

    @property
    def app(self):
        return TelegramIOAdapter._shared_app

    def setup(self, agent):
        pass

    async def send_message(self, text: str):
        if not self.app or not self.current_chat_id:
            return
        await self.app.bot.send_message(chat_id=self.current_chat_id, text=text)

    async def before_handle_output(self) -> bool:
        await self.chat_id_event.wait()
        return bool(self.app and self.current_chat_id)

    async def start(self):
        if not self.token:
            logger.error(
                "TELEGRAM_BOT_TOKEN is not set. Telegram adapter will not start."
            )
            return

        async with TelegramIOAdapter._app_lock:
            if TelegramIOAdapter._shared_app is None:
                app = Application.builder().token(self.token).build()
                app.add_handler(CommandHandler("start", self.shared_start_command))
                app.add_handler(
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, self.shared_handle_message
                    )
                )

                await app.initialize()
                await app.start()
                if app.updater:
                    await app.updater.start_polling(drop_pending_updates=True)
                TelegramIOAdapter._shared_app = app

        asyncio.create_task(self.process_events())

    @classmethod
    async def shared_start_command(
        cls, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not update.effective_chat or not update.message:
            return
        chat_id = update.effective_chat.id
        allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if allowed_chat_id and chat_id != int(allowed_chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        for adapter in cls._adapters:
            adapter.current_chat_id = chat_id
            adapter.chat_id_event.set()

        await update.message.reply_text("Agent is ready. Send a message to start.")

    @classmethod
    async def shared_handle_message(
        cls, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not update.effective_chat or not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if allowed_chat_id and chat_id != int(allowed_chat_id):
            return

        for adapter in cls._adapters:
            adapter.current_chat_id = chat_id
            adapter.chat_id_event.set()

        await cls._shared_input_queue.put(update.message.text)
