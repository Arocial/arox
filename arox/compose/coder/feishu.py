import asyncio
import json
import logging
import os

import lark_oapi as lark
from anyio import to_thread

from arox.ui.bot_base import BotIOAdapter
from arox.ui.io import AdapterIOInterface

logger = logging.getLogger(__name__)


class FeishuIOAdapter(BotIOAdapter):
    _ws_client: lark.ws.Client | None = None
    _lark_client: lark.Client | None = None
    _app_lock = asyncio.Lock()
    _adapters = []
    _shared_input_queue = asyncio.Queue()

    def __init__(self, adapter_io: AdapterIOInterface):
        super().__init__(adapter_io)
        self.app_id = os.environ.get("FEISHU_APP_ID")
        self.app_secret = os.environ.get("FEISHU_APP_SECRET")
        self.allowed_chat_id = os.environ.get("FEISHU_CHAT_ID")

        self.current_chat_id = self.allowed_chat_id
        self.input_queue = FeishuIOAdapter._shared_input_queue

        FeishuIOAdapter._adapters.append(self)

    def setup(self, agent):
        pass

    async def send_message(self, text: str):
        if not self.current_chat_id:
            return

        client = FeishuIOAdapter._lark_client
        if client is None or client.im is None:
            return

        request = (
            lark.im.v1.CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                lark.im.v1.CreateMessageRequestBody.builder()
                .receive_id(self.current_chat_id)
                .msg_type("text")
                .content(json.dumps({"text": text}))
                .build()
            )
            .build()
        )

        response = await client.im.v1.message.acreate(request)
        if not response.success():
            logger.error(
                f"client.im.v1.message.acreate failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            )

    async def before_handle_output(self) -> bool:
        return bool(self.current_chat_id)

    async def start(self):
        if not self.app_id or not self.app_secret:
            logger.error(
                "FEISHU_APP_ID or FEISHU_APP_SECRET is not set. Feishu adapter will not start."
            )
            return

        async with FeishuIOAdapter._app_lock:
            if FeishuIOAdapter._lark_client is None:
                FeishuIOAdapter._lark_client = (
                    lark.Client.builder()
                    .app_id(self.app_id)
                    .app_secret(self.app_secret)
                    .build()
                )

            if FeishuIOAdapter._ws_client is None:

                def do_p2_im_message_receive_v1(
                    data: lark.im.v1.P2ImMessageReceiveV1,
                ) -> None:
                    logger.info(f"Received Feishu message event: {data}")
                    if data.event and data.event.message:
                        msg = data.event.message
                        if msg.message_type == "text" and msg.content:
                            try:
                                text = json.loads(msg.content).get("text", "")
                                if text and msg.chat_id:
                                    logger.info(
                                        f"Processing text message from {msg.chat_id}: {text}"
                                    )
                                    asyncio.create_task(
                                        self.handle_user_message(msg.chat_id, text)
                                    )
                            except json.JSONDecodeError:
                                pass

                event_handler = (
                    lark.EventDispatcherHandler.builder("", "")
                    .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
                    .build()
                )

                wsclient = lark.ws.Client(
                    self.app_id,
                    self.app_secret,
                    event_handler=event_handler,
                    log_level=lark.LogLevel.INFO,
                )

                FeishuIOAdapter._ws_client = wsclient

                # wsclient did not provide a async variant.
                asyncio.create_task(
                    to_thread.run_sync(wsclient.start, abandon_on_cancel=True)
                )

        asyncio.create_task(self.process_events())

    async def handle_user_message(self, chat_id: str, text: str):
        if chat_id != self.allowed_chat_id:
            logger.error(f"chat id not allowed: {chat_id}.")
            return

        logger.info(f"Got user input: {text}")
        if self.input_queue:
            await self.input_queue.put(text)
