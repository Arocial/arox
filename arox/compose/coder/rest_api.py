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
        super().__init__(adapter_io)
        self.tool_ids = {}
        self.current_task = None
        self.read_lock = asyncio.Lock()

    async def start(self):
        pass

    async def run_cancellable(self, task):
        self.current_task = asyncio.create_task(task)
        try:
            await self.current_task
        except asyncio.CancelledError:
            logger.info("Task cancelled by client disconnect")
        finally:
            self.current_task = None

    async def drain_until_need_reply(self):
        try:
            while True:
                async with self.read_lock:
                    event = await self.adapter_io.adapter_receive()
                if isinstance(event, StepDoneEvent):
                    break
        except Exception as e:
            logger.error(f"Error draining events: {e}")

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
                async with self.read_lock:
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


class SuggestionItem(BaseModel):
    id: str
    value: str
    label: str
    description: str | None = None


class SuggestionResponse(BaseModel):
    items: list[SuggestionItem]


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
        self.app.get("/api/suggestions", response_model=SuggestionResponse)(
            self.suggestions
        )

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
        try:
            async for chunk in self.io_adapter.output_generator():
                logger.info(chunk)
                yield chunk
                if "data: [DONE]\n\n" == chunk:
                    break
        except asyncio.CancelledError:
            logger.info("Client disconnected, cancelling current task")
            if self.io_adapter.current_task:
                self.io_adapter.current_task.cancel()
            asyncio.create_task(self.io_adapter.drain_until_need_reply())
            raise

    async def suggestions(self, command: str | None = None, q: str | None = None):
        command_manager = self.composer.coder_agent.command_manager
        items = []

        if not command:
            for cmd_name, cmd_obj in command_manager.command_map.items():
                if q and q.lower() not in cmd_name.lower():
                    continue
                items.append(
                    SuggestionItem(
                        id=cmd_name,
                        value=f"/{cmd_name}",
                        label=f"/{cmd_name}",
                        description=cmd_obj.description,
                    )
                )
        else:
            args = q if q else ""
            completions = command_manager.get_completions(command, args)
            if completions:
                for idx, completion in enumerate(completions):
                    display_text = getattr(completion, "display_text", completion.text)
                    description = getattr(completion, "display_meta_text", None)
                    if not description:
                        description = None

                    items.append(
                        SuggestionItem(
                            id=f"comp-{command}-{idx}",
                            value=completion.text,
                            label=display_text,
                            description=description,
                        )
                    )

        return SuggestionResponse(items=items)

    def run(self):
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=8000)
