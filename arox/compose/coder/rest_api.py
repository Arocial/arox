import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai.ui.vercel_ai import request_types as vercel_ui_types

from arox.compose.coder.main import CoderComposer
from arox.ui.io import VercelStreamIOAdapter

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    messages: list[dict]


class CoderRestUI:
    def __init__(self):
        self.adapter = VercelStreamIOAdapter()
        self.composer = CoderComposer(lambda: self.adapter)
        self.app = FastAPI(lifespan=self.lifespan)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.post("/api/chat")(self.chat)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Start the composer in background
        asyncio.create_task(self.composer.run())
        try:
            yield
        finally:
            # We can't easily cancel the composer as it might be doing important things
            # But for now we just let it die with the process
            pass

    async def chat(self, request: ChatRequest):
        messages = request.messages
        if messages:
            last_message = vercel_ui_types.UIMessage.model_validate(messages[-1])
            if last_message.parts:
                part = last_message.parts[0]
                if isinstance(part, vercel_ui_types.TextUIPart):
                    content = part.text
                    logger.info(f"Got user input: {content}")
                    await self.adapter.submit_user_input(content)
                else:
                    logger.warning("Unsupported input type.")

        return StreamingResponse(
            self.response_generator(), media_type="text/event-stream"
        )

    async def response_generator(self):
        async for chunk in self.adapter.output_generator():
            logger.info(chunk)
            yield chunk
            if "data: [DONE]\n\n" == chunk:
                break

    def run(self):
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=8000)
