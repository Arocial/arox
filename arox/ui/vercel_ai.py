from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, override
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
    workspace: str | None = None
    session_id: str | None = None


class ComposerInfo(BaseModel):
    id: str
    workspace: str


class SessionInfo(BaseModel):
    id: str
    composer_name: str
    created_at: str
    updated_at: str
    metadata: dict


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self, adapter_io: AdapterIOInterface | None = None):
        super().__init__(adapter_io)
        self.tool_ids = {}
        self.current_tasks = {}
        self.read_lock = asyncio.Lock()
        self.coder_agents = {}
        self.event_queues = {}

    def setup(self, agent):
        self.coder_agents[agent.agent_io] = agent

    @override
    async def handle_event(self, adapter_io: AdapterIOInterface, event):
        queue = self.event_queues.setdefault(adapter_io, asyncio.Queue())
        await queue.put((adapter_io, event))

    async def run_cancellable(self, task, adapter_io: AdapterIOInterface):
        self.current_tasks[adapter_io] = asyncio.create_task(task)
        try:
            return await self.current_tasks[adapter_io]
        except asyncio.CancelledError:
            logger.info("Task cancelled by client disconnect")
        finally:
            self.current_tasks.pop(adapter_io, None)

    async def drain_until_need_reply(self, adapter_io: AdapterIOInterface):
        queue = self.event_queues.get(adapter_io)
        if not queue:
            return
        try:
            while True:
                _adapter_io, event = await queue.get()
                if isinstance(event, StepDoneEvent):
                    break
        except Exception as e:
            logger.error(f"Error draining events: {e}")

    def _format_event(self, adapter_io: AdapterIOInterface, event) -> list[str]:
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
                tool_ids = self.tool_ids.setdefault(adapter_io, {})
                tool_ids[index] = part.tool_call_id
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
                tool_ids = self.tool_ids.get(adapter_io, {})
                tool_id = tool_ids.get(index)
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

    async def output_generator(self, adapter_io: AdapterIOInterface):
        queue = self.event_queues.get(adapter_io)
        if not queue:
            yield "data: [DONE]\n\n"
            return
        try:
            while True:
                _adapter_io, event = await queue.get()
                if isinstance(event, StepDoneEvent):
                    yield "data: [DONE]\n\n"
                    break
                else:
                    formatted_events = self._format_event(adapter_io, event)
                    for fmt in formatted_events:
                        yield fmt
        except EndOfStream:
            yield "data: [DONE]\n\n"

    async def submit_user_input(self, adapter_io: AdapterIOInterface, text: str):
        from typing import cast

        from arox.ui.io import IOChannel

        io_channel = cast(IOChannel, adapter_io)
        if (
            io_channel.chat_input_event
            and not io_channel.chat_input_event.future.done()
        ):
            io_channel.chat_input_event.set_reply(json.loads(text))

    async def chat(self, adapter_io: AdapterIOInterface, request: ChatRequest):
        messages = request.messages
        if messages:
            last_message = vercel_ui_types.UIMessage.model_validate(messages[-1])
            if last_message.parts:
                part = last_message.parts[0]
                if isinstance(part, vercel_ui_types.TextUIPart):
                    content = part.text
                    logger.info(f"Got user input: {content}")
                    await self.submit_user_input(adapter_io, content)
                else:
                    logger.warning("Unsupported input type.")

        return StreamingResponse(
            self.response_generator(adapter_io), media_type="text/event-stream"
        )

    async def response_generator(self, adapter_io: AdapterIOInterface):
        try:
            async for chunk in self.output_generator(adapter_io):
                logger.info(chunk)
                yield chunk
                if "data: [DONE]\n\n" == chunk:
                    break
        except asyncio.CancelledError:
            logger.info("Client disconnected, cancelling current task")
            task = self.current_tasks.get(adapter_io)
            if task:
                task.cancel()
            asyncio.create_task(self.drain_until_need_reply(adapter_io))
            raise

    async def suggestions(
        self,
        adapter_io: AdapterIOInterface,
        command: str | None = None,
        q: str | None = None,
    ):
        agent = self.coder_agents.get(adapter_io)
        if not agent:
            return SuggestionResponse(items=[])

        command_manager = agent.command_manager
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
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        from contextlib import asynccontextmanager

        self.composer_name = composer_name
        self.config_files = config_files or []
        self.cli_args = cli_args or []
        self.host = host
        self.port = port
        self.composers: dict[str, Composer] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self.io_adapter = VercelStreamIOAdapter()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            # Cancel all running composer tasks on shutdown
            for task in self._tasks.values():
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self.app = FastAPI(lifespan=lifespan)
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
        self.app.get("/api/composers/{composer_id}/history")(self.history)
        self.app.get("/api/sessions", response_model=list[SessionInfo])(
            self.list_sessions
        )
        self.app.delete("/api/sessions/{session_id}")(self.delete_session)
        self.app.get("/api/health")(self.health)

    async def health(self):
        return {"status": "ok"}

    def _get_adapter_io(self, composer_id: str) -> AdapterIOInterface:
        composer = self.composers.get(composer_id)
        if not composer:
            raise HTTPException(status_code=404, detail="Composer not found")
        main_agent_name = composer.composer_config.main_agent
        return composer.io_channels[main_agent_name]

    async def create_composer(self, request: CreateComposerRequest):
        from arox.core.composer import Composer

        composer_id = uuid4().hex[:12]
        composer = Composer(
            self.composer_name,
            io_adapter=self.io_adapter,
            workspace=request.workspace,
            session_id=request.session_id,
            config_files=self.config_files,
            cli_args=self.cli_args,
        )
        self.composers[composer_id] = composer
        task = asyncio.create_task(self._run_composer(composer_id, composer))
        self._tasks[composer_id] = task
        return ComposerInfo(id=composer_id, workspace=str(composer.workspace))

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

    async def list_sessions(self):
        from arox.core.session import FileSessionStore

        store = FileSessionStore()
        sessions = await store.list_sessions(self.composer_name)
        return [
            SessionInfo(
                id=s.id,
                composer_name=s.composer_name,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
                metadata=s.metadata,
            )
            for s in sessions
        ]

    async def delete_session(self, session_id: str):
        from arox.core.session import FileSessionStore

        store = FileSessionStore()
        await store.delete_session(session_id)
        return {"status": "deleted"}

    async def delete_composer(self, composer_id: str):
        task = self._tasks.pop(composer_id, None)
        composer = self.composers.pop(composer_id, None)
        if not composer:
            raise HTTPException(status_code=404, detail="Composer not found")
        if task:
            task.cancel()
        return {"status": "deleted"}

    async def chat(self, composer_id: str, request: ChatRequest):
        adapter_io = self._get_adapter_io(composer_id)
        return await self.io_adapter.chat(adapter_io, request)

    async def suggestions(
        self, composer_id: str, command: str | None = None, q: str | None = None
    ):
        adapter_io = self._get_adapter_io(composer_id)
        return await self.io_adapter.suggestions(adapter_io, command, q)

    async def history(self, composer_id: str):
        composer = self.composers.get(composer_id)
        if not composer:
            raise HTTPException(status_code=404, detail="Composer not found")

        await composer.initialized.wait()

        if not composer.main_agent:
            return []

        from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

        messages = composer.main_agent.message_history
        ui_messages = VercelAIAdapter.dump_messages(messages)
        return [msg.model_dump(mode="json", exclude_none=True) for msg in ui_messages]

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="none")
        server = uvicorn.Server(config)

        # Start io_adapter
        asyncio.create_task(self.io_adapter.start())

        await server.serve()
