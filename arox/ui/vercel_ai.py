from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from arox.core.composer import Composer

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
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

from arox.core.chat import (
    ChatInputEvent,
    ChatInputReply,
    StepDoneEvent,
)
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)

logger = logging.getLogger(__name__)


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
        self.pending_inputs: dict[IOEndpoint, ChatInputEvent] = {}
        self.run_instances: dict[str, ComposerRun] = {}

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event):
        if isinstance(event, ChatInputEvent):
            self.pending_inputs[adapter_io] = event
        elif isinstance(event, StepDoneEvent):
            self.pending_inputs.pop(adapter_io, None)
        queue = self.event_queues.setdefault(adapter_io, asyncio.Queue())
        await queue.put((adapter_io, event))

    async def drain_until_need_reply(self, adapter_io: IOEndpoint):
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

    def _event_messages(self, adapter_io: IOEndpoint, event) -> list[dict]:
        messages: list[dict] = []

        if isinstance(event, PartStartEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                messages.append({"type": "text-start", "id": f"text_{index}"})
                if part.content:
                    messages.append(
                        {
                            "type": "text-delta",
                            "id": f"text_{index}",
                            "delta": part.content,
                        }
                    )

            elif isinstance(part, ThinkingPart):
                messages.append({"type": "reasoning-start", "id": f"reasoning_{index}"})
                if part.content:
                    messages.append(
                        {
                            "type": "reasoning-delta",
                            "id": f"reasoning_{index}",
                            "delta": part.content,
                        }
                    )

            elif isinstance(part, ToolCallPart):
                tool_ids = self.tool_ids.setdefault(adapter_io, {})
                tool_ids[index] = part.tool_call_id
                messages.append(
                    {
                        "type": "tool-input-start",
                        "toolCallId": part.tool_call_id,
                        "toolName": part.tool_name,
                    }
                )
                if part.args and isinstance(part.args, str):
                    messages.append(
                        {
                            "type": "tool-input-delta",
                            "toolCallId": part.tool_call_id,
                            "inputTextDelta": part.args,
                        }
                    )

        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            index = event.index

            if isinstance(delta, TextPartDelta):
                if delta.content_delta:
                    messages.append(
                        {
                            "type": "text-delta",
                            "id": f"text_{index}",
                            "delta": delta.content_delta,
                        }
                    )

            elif isinstance(delta, ThinkingPartDelta):
                if delta.content_delta:
                    messages.append(
                        {
                            "type": "reasoning-delta",
                            "id": f"reasoning_{index}",
                            "delta": delta.content_delta,
                        }
                    )

            elif isinstance(event.delta, ToolCallPartDelta):
                tool_ids = self.tool_ids.get(adapter_io, {})
                tool_id = tool_ids.get(index)
                if tool_id:
                    messages.append(
                        {
                            "type": "tool-input-delta",
                            "toolCallId": tool_id,
                            "inputTextDelta": delta.args_delta,
                        }
                    )

        elif isinstance(event, PartEndEvent):
            part = event.part
            index = event.index

            if isinstance(part, TextPart):
                messages.append({"type": "text-end", "id": f"text_{index}"})
            elif isinstance(part, ThinkingPart):
                messages.append({"type": "reasoning-end", "id": f"reasoning_{index}"})

        elif isinstance(event, FunctionToolCallEvent):
            part = event.part
            messages.append(
                {
                    "type": "tool-input-available",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "input": part.args,
                }
            )

        elif isinstance(event, FunctionToolResultEvent):
            messages.append(
                {
                    "type": "tool-output-available",
                    "toolCallId": event.tool_call_id,
                    "output": event.result.content,
                }
            )

        elif isinstance(event, FinalResultEvent):
            messages.append({"type": "finish"})

        elif isinstance(event, ChatInputEvent):
            messages.append(
                {"type": "data-input-request", "data": event.generate_request()}
            )

        elif isinstance(event, StepDoneEvent):
            messages.append({"type": "step-done"})

        return messages

    def _format_event(self, adapter_io: IOEndpoint, event) -> list[str]:
        return [
            f"data: {json.dumps(m)}\n\n"
            for m in self._event_messages(adapter_io, event)
        ]

    async def _apply_input(self, agent, payload: dict) -> dict:
        if payload.get("cancel"):
            cancel = getattr(agent, "cancel_foreground_task", None)
            if callable(cancel):
                cancel()
            return {"status": "cancelled"}

        cmd = payload.get("command")
        if cmd is not None:
            event = agent.command_manager.deserialize_event(cmd)
            if event is None:
                return {"status": "unknown_command"}
            reply = await agent.command_manager.execute(event)
            if reply.output:
                text_part = TextPart(content=reply.output)
                await self.handle_event(
                    agent.adapter_io,
                    PartStartEvent(part=text_part, index=-1),
                )
                await self.handle_event(
                    agent.adapter_io,
                    PartEndEvent(part=text_part, index=-1),
                )
            return {"status": "ok", "output": reply.output}

        reply = payload.get("reply")
        if reply is not None:
            req_id = reply.get("req_id")
            if not req_id:
                return {"status": "no_req_id"}
            deferred = reply.get("deferred_tools") or {}
            exception_input = reply.get("exception_input") or {}
            normal_input = reply.get("normal_input") or {}
            await agent.adapter_io.send(
                ChatInputReply(
                    req_id=req_id,
                    deferred_answers=dict(deferred),
                    user_input=normal_input.get("user_input"),
                    retry=bool(exception_input.get("retry", False)),
                )
            )
            self.pending_inputs.pop(agent.adapter_io, None)
            return {"status": "ok"}

        return {"status": "noop"}

    async def ws_handler(self, websocket: WebSocket, composer_id: str, agent_name: str):
        from fastapi import WebSocketDisconnect

        run_instance = self.run_instances.get(composer_id)
        if not run_instance:
            await websocket.close(code=4004, reason="composer not found")
            return
        composer = run_instance.composer
        agent = composer.all_agents().get(agent_name)
        if not agent:
            await websocket.close(code=4004, reason="agent not found")
            return

        adapter_io = agent.adapter_io
        queue = self.event_queues.setdefault(adapter_io, asyncio.Queue())

        await websocket.accept()

        async def pump_out():
            while True:
                _io, event = await queue.get()
                for msg in self._event_messages(adapter_io, event):
                    logger.info(f"WS OUT: {msg}")
                    await websocket.send_json(msg)

        async def pump_in():
            while True:
                payload = await websocket.receive_json()
                logger.info(f"WS IN: {payload}")

                if payload.get("resume"):
                    event = self.pending_inputs.get(adapter_io)
                    if event is not None:
                        for msg in self._event_messages(adapter_io, event):
                            await websocket.send_json(msg)
                else:
                    await self._apply_input(agent, payload)

        out_task = asyncio.create_task(pump_out())
        in_task = asyncio.create_task(pump_in())
        try:
            done, _ = await asyncio.wait(
                {out_task, in_task}, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    logger.exception("ws pump error", exc_info=exc)
        finally:
            out_task.cancel()
            in_task.cancel()
            await asyncio.gather(out_task, in_task, return_exceptions=True)

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
        command_manager = getattr(agent, "command_manager", None)
        if command_manager is None:
            return SuggestionResponse(items=[])
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
        self.app.websocket("/api/composers/{composer_id}/agents/{agent_name}/ws")(
            self.ws
        )
        self.app.get(
            "/api/composers/{composer_id}/agents/{agent_name}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)
        self.app.get("/api/composers/{composer_id}/agents/{agent_name}/state")(
            self.state
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

    async def ws(self, websocket: WebSocket, composer_id: str, agent_name: str):
        return await self.io_adapter.ws_handler(websocket, composer_id, agent_name)

    async def suggestions(
        self,
        composer_id: str,
        agent_name: str,
        command: str | None = None,
        q: str | None = None,
    ):
        return await self.io_adapter.suggestions(composer_id, agent_name, command, q)

    async def state(self, composer_id: str, agent_name: str):
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
        history = [
            msg.model_dump(mode="json", exclude_none=True) for msg in ui_messages
        ]

        return {"history": history}

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

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="auto")
        server = uvicorn.Server(config)

        await server.serve()
