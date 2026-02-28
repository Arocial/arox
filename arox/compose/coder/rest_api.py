import asyncio
import json
import logging
from contextlib import asynccontextmanager

from anyio import EndOfStream
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.ui.vercel_ai import request_types as vercel_ui_types

from arox.compose.coder.main import CoderComposer
from arox.ui.io import (
    AbstractIOAdapter,
    AdapterIOInterface,
    NeedReplyEvent,
    StepDoneEvent,
)

logger = logging.getLogger(__name__)


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface):
        self.adapter_io = adapter_io
        self.tool_ids = {}

    async def start(self):
        pass

    def _format_event(self, event) -> list[str]:
        events = []

        if isinstance(event, PartStartEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                events.append(
                    f"data: {json.dumps({'type': 'text-start', 'id': f'text_{index}'})}\n\n"
                )
                if part.content:
                    events.append(
                        f"data: {json.dumps({'type': 'text-delta', 'id': f'text_{index}', 'delta': part.content})}\n\n"
                    )

            elif isinstance(part, ThinkingPart):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-start', 'id': f'reasoning_{index}'})}\n\n"
                )
                if part.content:
                    events.append(
                        f"data: {json.dumps({'type': 'reasoning-delta', 'id': f'reasoning_{index}', 'delta': part.content})}\n\n"
                    )

            elif isinstance(part, ToolCallPart):
                self.tool_ids[index] = part.tool_call_id
                events.append(
                    f"data: {json.dumps({'type': 'tool-input-start', 'toolCallId': part.tool_call_id, 'toolName': part.tool_name})}\n\n"
                )
                if part.args and isinstance(part.args, str):
                    events.append(
                        f"data: {json.dumps({'type': 'tool-input-delta', 'toolCallId': part.tool_call_id, 'inputTextDelta': part.args})}\n\n"
                    )

        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            index = event.index

            if isinstance(delta, TextPartDelta):
                events.append(
                    f"data: {json.dumps({'type': 'text-delta', 'id': f'text_{index}', 'delta': delta.content_delta})}\n\n"
                )

            elif isinstance(delta, ThinkingPartDelta):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-delta', 'id': f'reasoning_{index}', 'delta': delta.content_delta})}\n\n"
                )

            elif isinstance(event.delta, ToolCallPartDelta):
                tool_id = self.tool_ids.get(index)
                if tool_id:
                    events.append(
                        f"data: {json.dumps({'type': 'tool-input-delta', 'toolCallId': tool_id, 'inputTextDelta': delta.args_delta})}\n\n"
                    )

        elif isinstance(event, PartEndEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                events.append(
                    f"data: {json.dumps({'type': 'text-end', 'id': f'text_{index}'})}\n\n"
                )
            elif isinstance(part, ThinkingPart):
                events.append(
                    f"data: {json.dumps({'type': 'reasoning-end', 'id': f'reasoning_{index}'})}\n\n"
                )

        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            events.append(
                f"data: {json.dumps({'type': 'tool-input-available', 'toolCallId': part.tool_call_id, 'toolName': part.tool_name, 'input': part.args})}\n\n"
            )

        elif isinstance(event, FunctionToolResultEvent):
            events.append(
                f"data: {json.dumps({'type': 'tool-output-available', 'toolCallId': event.tool_call_id, 'output': event.result.content})}\n\n"
            )

        elif isinstance(event, FinalResultEvent):
            events.append(f"data: {json.dumps({'type': 'finish'})}\n\n")

        return events

    async def output_generator(self):
        try:
            while True:
                event = await self.adapter_io.adapter_receive()
                if isinstance(event, NeedReplyEvent):
                    pass
                elif isinstance(event, StepDoneEvent):
                    yield "data: [DONE]\n\n"
                    break
                else:
                    formatted_events = self._format_event(event)
                    for fmt in formatted_events:
                        yield fmt
        except EndOfStream:
            yield "data: [DONE]\n\n"

    async def submit_user_input(self, text: str):
        await self.adapter_io.adapter_send(text)


class ChatRequest(BaseModel):
    messages: list[dict]


class CoderRestUI:
    def __init__(self):
        self.composer = CoderComposer(VercelStreamIOAdapter)
        self.io_adapter = self.composer.coder_adapter
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
        asyncio.create_task(self.composer.run())
        try:
            yield
        finally:
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
                    await self.io_adapter.submit_user_input(content)
                else:
                    logger.warning("Unsupported input type.")

        return StreamingResponse(
            self.response_generator(), media_type="text/event-stream"
        )

    async def response_generator(self):
        async for chunk in self.io_adapter.output_generator():
            logger.info(chunk)
            yield chunk
            if "data: [DONE]\n\n" == chunk:
                break

    def run(self):
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=8000)
