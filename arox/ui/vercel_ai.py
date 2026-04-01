from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from arox.core.composer import Composer

from anyio import EndOfStream
from fastapi import FastAPI, HTTPException
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

from arox.ui.io import (
    AbstractIOAdapter,
    AdapterIOInterface,
    ChatInputEvent,
    StepDoneEvent,
)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    messages: list[dict]


class SuggestionItem(BaseModel):
    id: str
    value: str
    label: str
    description: str | None = None


class SuggestionResponse(BaseModel):
    items: list[SuggestionItem]


class CreateComposerRequest(BaseModel):
    workspace: str


class ComposerInfo(BaseModel):
    id: str
    workspace: str


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface | None = None):
        super().__init__(adapter_io)
        self.tool_ids = {}
        self.current_task = None
        self.read_lock = asyncio.Lock()
        self.coder_agent = None
        self.event_queue = asyncio.Queue()

    def setup(self, agent):
        self.coder_agent = agent

    async def start(self):
        import anyio

        async def process_io(adapter_io):
            try:
                while True:
                    event = await adapter_io.adapter_receive()
                    await self.event_queue.put((adapter_io, event))
            except EndOfStream:
                pass

        async with anyio.create_task_group() as tg:
            for adapter_io in self.adapter_ios:
                tg.start_soon(process_io, adapter_io)

    async def run_cancellable(self, task):
        self.current_task = asyncio.create_task(task)
        try:
            return await self.current_task
        except asyncio.CancelledError:
            logger.info("Task cancelled by client disconnect")
        finally:
            self.current_task = None

    async def drain_until_need_reply(self):
        try:
            while True:
                _adapter_io, event = await self.event_queue.get()
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
                if delta.content_delta:
                    events.append(
                        f"data: {json.dumps({'type': 'text-delta', 'id': f'text_{index}', 'delta': delta.content_delta})}\n\n"
                    )

            elif isinstance(delta, ThinkingPartDelta):
                if delta.content_delta:
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

        elif isinstance(event, ChatInputEvent):
            events.append(
                f"data: {json.dumps({'type': 'data-input-request', 'data': event.generate_request()})}\n\n"
            )

        return events

    async def output_generator(self):
        try:
            while True:
                _adapter_io, event = await self.event_queue.get()
                if isinstance(event, StepDoneEvent):
                    yield "data: [DONE]\n\n"
                    break
                else:
                    formatted_events = self._format_event(event)
                    for fmt in formatted_events:
                        yield fmt
        except EndOfStream:
            yield "data: [DONE]\n\n"

    async def submit_user_input(self, text: str):
        from typing import cast

        from arox.ui.io import IOChannel

        for adapter_io in self.adapter_ios:
            io_channel = cast(IOChannel, adapter_io)
            if (
                io_channel.chat_input_event
                and not io_channel.chat_input_event.future.done()
            ):
                io_channel.chat_input_event.set_reply(json.loads(text))
                break

    async def chat(self, request: ChatRequest):
        messages = request.messages
        if messages:
            last_message = vercel_ui_types.UIMessage.model_validate(messages[-1])
            if last_message.parts:
                part = last_message.parts[0]
                if isinstance(part, vercel_ui_types.TextUIPart):
                    content = part.text
                    logger.info(f"Got user input: {content}")
                    await self.submit_user_input(content)
                else:
                    logger.warning("Unsupported input type.")

        return StreamingResponse(
            self.response_generator(), media_type="text/event-stream"
        )

    async def response_generator(self):
        try:
            async for chunk in self.output_generator():
                logger.info(chunk)
                yield chunk
                if "data: [DONE]\n\n" == chunk:
                    break
        except asyncio.CancelledError:
            logger.info("Client disconnected, cancelling current task")
            if self.current_task:
                self.current_task.cancel()
            asyncio.create_task(self.drain_until_need_reply())
            raise

    async def suggestions(self, command: str | None = None, q: str | None = None):
        if not self.coder_agent:
            return SuggestionResponse(items=[])

        command_manager = self.coder_agent.command_manager
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


class VercelStreamServer:
    def __init__(
        self,
        composer_name: str,
        config_files: list[str | Path] | None = None,
        cli_args: list[str] | None = None,
    ):
        self.composer_name = composer_name
        self.config_files = config_files or []
        self.cli_args = cli_args or []
        self.composers: dict[str, Composer] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.post("/api/composers", response_model=ComposerInfo)(
            self.create_composer
        )
        self.app.get("/api/composers", response_model=list[ComposerInfo])(
            self.list_composers
        )
        self.app.delete("/api/composers/{composer_id}")(self.delete_composer)
        self.app.post("/api/composers/{composer_id}/chat")(self.chat)
        self.app.get(
            "/api/composers/{composer_id}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)

    def _get_adapter(self, composer_id: str) -> VercelStreamIOAdapter:
        composer = self.composers.get(composer_id)
        if not composer:
            raise HTTPException(status_code=404, detail="Composer not found")
        return composer.io_adapter

    async def create_composer(self, request: CreateComposerRequest):
        from arox.core.composer import Composer

        composer_id = uuid4().hex[:12]
        composer = Composer(
            self.composer_name,
            workspace=request.workspace,
            config_files=self.config_files,
            cli_args=self.cli_args,
        )
        self.composers[composer_id] = composer
        task = asyncio.create_task(self._run_composer(composer_id, composer))
        self._tasks[composer_id] = task
        return ComposerInfo(id=composer_id, workspace=request.workspace)

    async def _run_composer(self, composer_id: str, composer):
        try:
            await composer.run()
        except asyncio.CancelledError:
            logger.info(f"Composer {composer_id} cancelled")
        except Exception:
            logger.exception(f"Composer {composer_id} error")
        finally:
            self.composers.pop(composer_id, None)
            self._tasks.pop(composer_id, None)

    async def list_composers(self):
        return [
            ComposerInfo(id=cid, workspace=str(c.workspace))
            for cid, c in self.composers.items()
        ]

    async def delete_composer(self, composer_id: str):
        task = self._tasks.pop(composer_id, None)
        composer = self.composers.pop(composer_id, None)
        if not composer:
            raise HTTPException(status_code=404, detail="Composer not found")
        if task:
            task.cancel()
        return {"status": "deleted"}

    async def chat(self, composer_id: str, request: ChatRequest):
        adapter = self._get_adapter(composer_id)
        return await adapter.chat(request)

    async def suggestions(
        self, composer_id: str, command: str | None = None, q: str | None = None
    ):
        adapter = self._get_adapter(composer_id)
        return await adapter.suggestions(command, q)

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000, ws="none")
        server = uvicorn.Server(config)
        await server.serve()
