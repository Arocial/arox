from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, override

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

from arox.core.chat import ChatAgent, ChatInputEvent, StepDoneEvent
from arox.core.io import (
    AbstractIOAdapter,
    AdapterIOEndpoint,
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
    main_agent: str
    subagents: list[str]


class SessionInfo(BaseModel):
    id: str
    composer_name: str
    created_at: str
    updated_at: str
    metadata: dict


from dataclasses import dataclass
from enum import Enum


class ComposerTaskStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ComposerRun:
    composer: Composer
    task: asyncio.Task | None = None
    status: ComposerTaskStatus = ComposerTaskStatus.RUNNING
    error: Exception | None = None


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.tool_ids = {}
        self.read_lock = asyncio.Lock()
        self.event_queues = {}
        self.run_instances: dict[str, ComposerRun] = {}

    @override
    async def handle_event(self, adapter_io: AdapterIOEndpoint, event):
        queue = self.event_queues.setdefault(adapter_io, asyncio.Queue())
        await queue.put((adapter_io, event))

    async def drain_until_need_reply(self, adapter_io: AdapterIOEndpoint):
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

    def _format_event(self, adapter_io: AdapterIOEndpoint, event) -> list[str]:
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

    async def output_generator(self, adapter_io: AdapterIOEndpoint):
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

    async def submit_user_input(self, agent, text: str):
        if (
            hasattr(agent, "current_chat_input_event")
            and agent.current_chat_input_event
            and not agent.current_chat_input_event.future.done()
        ):
            agent.current_chat_input_event.set_reply(json.loads(text))

    async def chat(self, composer_id: str, agent_name: str, request: ChatRequest):
        run_instance = self.run_instances.get(composer_id)
        if not run_instance:
            raise HTTPException(
                status_code=404, detail=f"Composer {composer_id} not found."
            )
        composer = run_instance.composer
        agent = composer.all_agents().get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent {agent_name} not found."
            )
        adapter_io = agent.adapter_io

        messages = request.messages
        if messages:
            last_message = vercel_ui_types.UIMessage.model_validate(messages[-1])
            if last_message.parts:
                part = last_message.parts[0]
                if isinstance(part, vercel_ui_types.TextUIPart):
                    content = part.text
                    logger.info(f"Got user input: {content}")
                    await self.submit_user_input(agent, content)
                else:
                    logger.warning("Unsupported input type.")

        return StreamingResponse(
            self.response_generator(agent, adapter_io),
            media_type="text/event-stream",
        )

    async def response_generator(self, agent, adapter_io: AdapterIOEndpoint):
        try:
            async for chunk in self.output_generator(adapter_io):
                logger.info(chunk)
                yield chunk
                if "data: [DONE]\n\n" == chunk:
                    break
        except asyncio.CancelledError:
            logger.info("Client disconnected, cancelling current task")
            agent.cancel_foreground_task()
            asyncio.create_task(self.drain_until_need_reply(adapter_io))
            raise

    async def suggestions(
        self,
        composer_id: str,
        agent_name: str,
        command: str | None = None,
        q: str | None = None,
    ):
        run_instance = self.run_instances.get(composer_id)
        if not run_instance:
            raise HTTPException(
                status_code=404, detail=f"Composer {composer_id} not found."
            )
        composer = run_instance.composer
        agent = composer.all_agents().get(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent {agent_name} not found."
            )
        if not isinstance(agent, ChatAgent):
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
        self.io_adapter = VercelStreamIOAdapter()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            # Cancel all running composer tasks on shutdown
            tasks = [
                r.task
                for r in self.io_adapter.run_instances.values()
                if r.task and not r.task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
        self.app.post("/api/composers/{composer_id}/agents/{agent_name}/chat")(
            self.chat
        )
        self.app.get(
            "/api/composers/{composer_id}/agents/{agent_name}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)
        self.app.get("/api/composers/{composer_id}/agents/{agent_name}/history")(
            self.history
        )
        self.app.get("/api/sessions", response_model=list[SessionInfo])(
            self.list_sessions
        )
        self.app.delete("/api/sessions/{session_id}")(self.delete_session)
        self.app.get("/api/health")(self.health)

    async def health(self):
        return {"status": "ok"}

    async def create_composer(self, request: CreateComposerRequest):
        from arox.core.composer import Composer

        composer = Composer(
            self.composer_name,
            io_adapter=self.io_adapter,
            workspace=request.workspace,
            session_id=request.session_id,
            config_files=self.config_files,
            cli_args=self.cli_args,
        )
        run_instance = ComposerRun(composer=composer)
        self.io_adapter.run_instances[composer.id] = run_instance

        task = asyncio.create_task(composer.run())
        run_instance.task = task

        def on_task_done(t: asyncio.Task):
            try:
                t.result()
                run_instance.status = ComposerTaskStatus.STOPPED
                logger.info(f"Composer {composer.id} finished normally.")
            except asyncio.CancelledError:
                run_instance.status = ComposerTaskStatus.CANCELLED
                logger.info(f"Composer {composer.id} was cancelled.")
            except Exception as e:
                run_instance.status = ComposerTaskStatus.ERROR
                run_instance.error = e
                logger.exception(f"Composer {composer.id} crashed with error.")

        task.add_done_callback(on_task_done)

        return ComposerInfo(
            id=composer.id,
            workspace=str(composer.workspace),
            main_agent=composer.main_agent.name,
            subagents=list(composer.subagents.keys()),
        )

    async def list_composers(self):
        return [
            ComposerInfo(
                id=cid,
                workspace=str(r.composer.workspace),
                main_agent=r.composer.main_agent.name,
                subagents=list(r.composer.subagents.keys()),
            )
            for cid, r in self.io_adapter.run_instances.items()
        ]

    async def delete_composer(self, composer_id: str):
        run_instance = self.io_adapter.run_instances.pop(composer_id, None)
        if not run_instance:
            raise HTTPException(status_code=404, detail="Composer not found")
        if run_instance.task and not run_instance.task.done():
            run_instance.task.cancel()
        return {"status": "deleted"}

    async def chat(self, composer_id: str, agent_name: str, request: ChatRequest):
        return await self.io_adapter.chat(composer_id, agent_name, request)

    async def suggestions(
        self,
        composer_id: str,
        agent_name: str,
        command: str | None = None,
        q: str | None = None,
    ):
        return await self.io_adapter.suggestions(composer_id, agent_name, command, q)

    async def history(self, composer_id: str, agent_name: str):
        run_instance = self.io_adapter.run_instances.get(composer_id)
        if not run_instance:
            raise HTTPException(status_code=404, detail="Composer not found")

        composer = run_instance.composer
        agent = composer.all_agents().get(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

        messages = agent.message_history
        ui_messages = VercelAIAdapter.dump_messages(messages)
        return [msg.model_dump(mode="json", exclude_none=True) for msg in ui_messages]

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

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="none")
        server = uvicorn.Server(config)

        await server.serve()
