from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, override
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_ai import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, UIMessage
from starlette.types import ASGIApp, Receive, Scope, Send

from arox.core.chat import (
    ChatInputReply,
    ChatInputRequest,
    ChatServeDriver,
    StepDoneEvent,
)
from arox.core.completion import parse_request
from arox.core.config import ConfigLoader
from arox.core.io import AbstractIOAdapter, IOEndpoint
from arox.core.runner import ServeRunner, TaskRunner
from arox.core.session import AgentSession
from arox.core.types import ServerIdMapping, SessionTreeUpdate, UserMessageEvent

if TYPE_CHECKING:
    from arox.core.session import SessionManager


async def build_session_view(session: AgentSession) -> SessionView:
    child_sessions = (
        await session.manager.children_of(session) if session.manager else []
    )
    children = [
        await build_session_view(child)
        for child in child_sessions
        if isinstance(child, AgentSession)
    ]
    return SessionView(
        id=session.id,
        path=session.path,
        agent_name=session.agent_name,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        workspace=session.workspace,
        metadata=session.metadata,
        active=session.is_active,
        task_name=session.task_name,
        target=session.target,
        children=children,
    )


class TokenAuthASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        expected_token = os.environ.get("AROX_API_TOKEN")
        if not expected_token:
            return await self.app(scope, receive, send)

        if scope["path"] == "/api/health":
            return await self.app(scope, receive, send)

        token = None
        headers = dict(scope.get("headers", []))

        auth_header = headers.get(b"authorization")
        if auth_header and auth_header.startswith(b"Bearer "):
            token = auth_header[7:].decode("utf-8")
        else:
            query_string = scope.get("query_string", b"").decode("utf-8")
            qs = parse_qs(query_string)
            if "token" in qs:
                token = qs["token"][0]

        if not token or not secrets.compare_digest(token, expected_token):
            if scope["type"] == "http":
                await send(
                    {"type": "http.response.start", "status": 401, "headers": []}
                )
                await send({"type": "http.response.body", "body": b""})
            elif scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4001})
            return

        return await self.app(scope, receive, send)


logger = logging.getLogger(__name__)


def build_state_history(messages: list) -> list[dict]:
    from arox.core.types import USER_INPUT_ID_KEY

    wrapped_messages = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if isinstance(part.content, str):
                        new_parts.append(part)
                        continue

                    content = (
                        part.content
                        if isinstance(part.content, (list, tuple))
                        else [part.content]
                    )
                    new_content = []
                    for item in content:
                        if isinstance(item, TextContent) and isinstance(
                            item.metadata, dict
                        ):
                            input_id = item.metadata.get(USER_INPUT_ID_KEY)
                            if input_id:
                                wrapped = json.dumps(
                                    {"_arox_id": str(input_id), "text": item.content}
                                )
                                new_content.append(
                                    dataclasses.replace(item, content=wrapped)
                                )
                                continue
                        new_content.append(item)
                    new_parts.append(dataclasses.replace(part, content=new_content))
                else:
                    new_parts.append(part)
            # Ensure we don't drop original ModelRequest metadata or other fields
            wrapped_messages.append(dataclasses.replace(msg, parts=new_parts))
        else:
            wrapped_messages.append(msg)

    ui_messages = VercelAIAdapter.dump_messages(wrapped_messages)
    # `by_alias` to serialize keys as camel case, which assistant-ui
    # recognizes. See `pydantic_ai/ui/vercel_ai/_models.py:CamelBaseModel`
    history = [
        msg.model_dump(mode="json", exclude_none=True, by_alias=True)
        for msg in ui_messages
    ]

    # Hoist it to message-level ``metadata.custom`` — the only slot @assistant-ui
    # preserves for user messages — so the client reads the fork anchor straight
    # off the message, the same shape it gets live via ``data-user-turn``.
    for msg in history:
        if msg.get("role") != "user":
            continue
        for part in msg.get("parts", []):
            if part.get("type") == "text":
                text = part.get("text", "")
                if text.startswith("{") and '"_arox_id"' in text:
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict) and "_arox_id" in data:
                            part["text"] = data.get("text", "")
                            custom = msg.setdefault("metadata", {}).setdefault(
                                "custom", {}
                            )
                            custom[USER_INPUT_ID_KEY] = data["_arox_id"]
                    except json.JSONDecodeError:
                        pass

    return history


class SuggestionItem(BaseModel):
    id: str
    value: str
    label: str
    description: str | None = None


class SuggestionResponse(BaseModel):
    items: list[SuggestionItem]


class CreateSessionRequest(BaseModel):
    workspace: str | None = None


class SessionView(BaseModel):
    id: str
    path: list[str]
    agent_name: str
    created_at: str
    updated_at: str
    workspace: str | None
    metadata: dict
    active: bool
    task_name: str | None
    target: str | None
    children: list["SessionView"] = Field(default_factory=list)


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.event_queues = {}
        self.pending_inputs: dict[IOEndpoint, ChatInputRequest] = {}
        self._pydanticai_vercel_streams: dict[IOEndpoint, VercelAIEventStream] = {}

    @override
    async def handle_event(self, adapter_ep: IOEndpoint, event):
        if isinstance(event, ChatInputRequest):
            self.pending_inputs[adapter_ep] = event
        elif isinstance(event, StepDoneEvent):
            self.pending_inputs.pop(adapter_ep, None)
        queue = self.event_queues.setdefault(adapter_ep, asyncio.Queue())
        await queue.put((adapter_ep, event))

    async def _to_ui_messages(
        self,
        adapter_ep: IOEndpoint,
        event,
        root_session: AgentSession,
    ) -> list[dict]:
        messages: list[dict] = []

        if isinstance(
            event,
            (
                PartStartEvent,
                PartDeltaEvent,
                PartEndEvent,
                FunctionToolCallEvent,
                FunctionToolResultEvent,
            ),
        ):
            stream = self._pydanticai_vercel_streams.setdefault(
                adapter_ep,
                VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
            )
            async for chunk in stream.handle_event(event):
                messages.append(chunk.model_dump(by_alias=True, exclude_none=True))

        elif isinstance(event, FinalResultEvent):
            messages.append({"type": "finish"})

        elif isinstance(event, ChatInputRequest):
            if event.pending_exception:
                messages.append(
                    {
                        "type": "error",
                        "errorText": f"{type(event.pending_exception).__name__}: {event.pending_exception}",
                    }
                )
            messages.append(
                {
                    "type": "cmd-input-request",
                    "req_id": event.req_id,
                    "normal_input": event.request_normal_input,
                }
            )
            # Emit an explicit stream close signal to let the frontend decouple
            # stream management from business logic.
            messages.append({"type": "stream-close"})

        elif isinstance(event, StepDoneEvent):
            messages.append({"type": "step-done"})

        elif isinstance(event, ServerIdMapping):
            frame = {
                "type": "cmd-user-turn",
                "server_message_id": event.server_message_id,
                "client_message_id": event.client_message_id,
            }
            messages.append(frame)

        elif isinstance(event, UserMessageEvent):
            input_content = event.user_input.input_content
            if input_content is not None:
                request = ModelRequest(parts=[UserPromptPart(content=input_content)])
                history = build_state_history([request])
                if history:
                    message = history[0]
                    message["id"] = event.user_input.server_message_id
                    messages.append({"type": "cmd-user-message", "message": message})

        elif isinstance(event, SessionTreeUpdate):
            if event.session_id == root_session.id:
                messages.append(
                    {
                        "type": "cmd-session-tree",
                        **((await build_session_view(root_session)).model_dump()),
                    }
                )

        return messages

    async def _render_command_output(self, target, output: str | None) -> None:
        """Stream a command's text output through the normal event pipeline."""
        if not output:
            return
        await target.agent_ep.send(output)

    async def _apply_input(self, target, payload: dict) -> dict:
        if payload.get("cancel"):
            runner = target.session.runner
            if isinstance(runner, ServeRunner):
                await runner.cancel_current_interaction()
            elif isinstance(runner, TaskRunner):
                task = runner.task
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            return {"status": "cancelled"}

        cmd = payload.get("command")
        if cmd is not None:
            event = target.command_manager.deserialize_event(cmd)
            if event is None:
                return {"status": "unknown_command"}
            reply = await target.command_manager.execute(event)
            await self._render_command_output(target, reply.output)
            return {"status": "ok", "output": reply.output}

        reply = payload.get("reply")
        if reply is not None:
            reply_message = UIMessage.model_validate(reply)
            if reply_message.metadata:
                custom_metadata = reply_message.metadata.get("custom", {})
            else:
                custom_metadata = {}
            reply_metadata = custom_metadata.get("chatInputEventResult")
            if not isinstance(reply_metadata, dict):
                return {"status": "error", "message": "no chatInputEventResult"}

            user_msg = VercelAIAdapter.load_messages([reply_message])
            parts = user_msg[0].parts
            assert len(parts) == 1
            from pydantic_ai.messages import UserPromptPart

            part = parts[0]
            assert isinstance(part, UserPromptPart)
            input_content = part.content
            chat_input_reply = ChatInputReply(
                req_id=reply_metadata["req_id"],
                input_content=input_content,
                client_message_id=reply_metadata.get("client_message_id"),
            )

            self.pending_inputs.pop(target.adapter_ep, None)
            await target.adapter_ep.send(chat_input_reply)
            return {"status": "ok"}

        return {"status": "noop"}

    async def ws_handler(
        self,
        websocket: WebSocket,
        root_session: AgentSession,
        target_session: AgentSession,
    ):
        from fastapi import WebSocketDisconnect

        runtime = (
            target_session.runner.runtime if target_session.runner is not None else None
        )
        if runtime is None:
            await websocket.close(code=4009, reason="session runtime is not active")
            return
        adapter_ep = runtime.adapter_ep
        queue = self.event_queues.setdefault(adapter_ep, asyncio.Queue())

        await websocket.accept()

        async def pump_out():
            while True:
                _io, event = await queue.get()
                for msg in await self._to_ui_messages(adapter_ep, event, root_session):
                    msg_str = str(msg)
                    if len(msg_str) > 1024:
                        msg_str = msg_str[:1024] + "... (truncated)"
                    logger.info(f"WS OUT: {msg_str}")
                    await websocket.send_json(msg)

        async def pump_in():
            while True:
                payload = await websocket.receive_json()
                payload_str = str(payload)
                if len(payload_str) > 1024:
                    payload_str = payload_str[:1024] + "... (truncated)"
                logger.info(f"WS IN: {payload_str}")

                if payload.get("resume"):
                    event = self.pending_inputs.get(adapter_ep)
                    if event is not None:
                        for msg in await self._to_ui_messages(
                            adapter_ep, event, root_session
                        ):
                            await websocket.send_json(msg)
                    await websocket.send_json({"type": "ack", "status": "ok"})
                else:
                    ack = await self._apply_input(runtime, payload)
                    await websocket.send_json({"type": "ack", **ack})

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
        target,
        command: str | None = None,
        q: str | None = None,
    ) -> SuggestionResponse:
        if command:
            text = f"/{command} {q or ''}"
        else:
            text = f"/{q or ''}"
        req = parse_request(text, runtime=target)
        results = await target.command_manager.completion_router.complete(req)

        items = [
            SuggestionItem(
                id=f"{it.group or 'item'}:{it.value}",
                value=it.value,
                label=it.label or it.value,
                description=it.description,
            )
            for it in results
        ]
        return SuggestionResponse(items=items)


class VercelStreamServer:
    def __init__(
        self,
        session_manager: "SessionManager",
        config_loader: ConfigLoader,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        from contextlib import asynccontextmanager

        self.config_loader = config_loader
        self.host = host
        self.port = port
        self.io_adapter = VercelStreamIOAdapter()
        self.session_manager = session_manager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Stop all session runtimes before closing their shared IO adapter.
            async with self.io_adapter, self.session_manager:
                yield

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_middleware(TokenAuthASGIMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.get("/api/sessions", response_model=list[SessionView])(
            self.list_sessions
        )
        self.app.post("/api/sessions", response_model=SessionView)(self.create_session)
        self.app.get("/api/sessions/{session_id}", response_model=SessionView)(
            self.get_session
        )
        self.app.post("/api/sessions/{session_id}/start", response_model=SessionView)(
            self.start_session
        )
        self.app.post("/api/sessions/{session_id}/stop", response_model=SessionView)(
            self.stop_session
        )
        self.app.delete("/api/sessions/{session_id}")(self.delete_session)
        self.app.websocket(
            "/api/sessions/{root_session_id}/nodes/{target_session_id}/ws"
        )(self.ws)
        self.app.get(
            "/api/sessions/{root_session_id}/nodes/{target_session_id}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)
        self.app.get("/api/sessions/{root_session_id}/nodes/{target_session_id}/state")(
            self.state
        )
        self.app.get("/api/health")(self.health)

    async def health(self):
        return {"status": "ok"}

    async def _resolve_root_session(self, session_id: str) -> AgentSession:
        loaded = await self.session_manager.resolve(session_id)
        session = loaded if isinstance(loaded, AgentSession) else None
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session

    async def _resolve_child_session(
        self, root: AgentSession, target_session_id: str
    ) -> AgentSession:
        found = await self.session_manager.find(root, target_session_id)
        if isinstance(found, AgentSession):
            return found
        raise HTTPException(status_code=404, detail="Session node not found.")

    async def create_session(self, request: CreateSessionRequest) -> SessionView:
        main_agent_name = self.config_loader.current_config.app.main_agent
        session = AgentSession(
            agent_name=main_agent_name,
            workspace=str(Path(request.workspace).absolute())
            if request.workspace
            else None,
            manager=self.session_manager,
        )
        await self.session_manager.persist(session)
        return await self.start_session(session.id)

    async def _supervise_runner(
        self,
        runner: ServeRunner,
        serve_task: asyncio.Task[None],
    ) -> None:
        try:
            await asyncio.gather(serve_task, return_exceptions=True)
        finally:
            await runner.stop_runtime()

    async def start_session(self, session_id: str) -> SessionView:
        session = await self._resolve_root_session(session_id)
        if session.is_active:
            raise HTTPException(status_code=409, detail="Session is already active.")
        runner = ServeRunner(
            session, self.config_loader, self.io_adapter, ChatServeDriver()
        )
        await runner.start_runtime()
        serve_task = runner.run()
        asyncio.create_task(
            self._supervise_runner(runner, serve_task),
            name=f"session-supervisor:{session.id}",
        )
        return await build_session_view(session)

    async def stop_session(self, session_id: str) -> SessionView:
        session = await self._resolve_root_session(session_id)
        await self.session_manager.stop_tree(session)
        return await build_session_view(session)

    async def get_session(self, session_id: str) -> SessionView:
        return await build_session_view(await self._resolve_root_session(session_id))

    async def ws(
        self,
        websocket: WebSocket,
        root_session_id: str,
        target_session_id: str,
    ):
        try:
            root = await self._resolve_root_session(root_session_id)
            target = await self._resolve_child_session(root, target_session_id)
        except HTTPException as exc:
            await websocket.close(code=4004, reason=str(exc.detail))
            return
        return await self.io_adapter.ws_handler(websocket, root, target)

    async def suggestions(
        self,
        root_session_id: str,
        target_session_id: str,
        command: str | None = None,
        q: str | None = None,
    ):
        root = await self._resolve_root_session(root_session_id)
        target = await self._resolve_child_session(root, target_session_id)
        runtime = target.runner.runtime if target.runner is not None else None
        if runtime is None:
            raise HTTPException(
                status_code=409, detail="Session runtime is not active."
            )
        return await self.io_adapter.suggestions(runtime, command, q)

    async def state(self, root_session_id: str, target_session_id: str):
        root = await self._resolve_root_session(root_session_id)
        session = await self._resolve_child_session(root, target_session_id)

        from pydantic_ai import ModelRequest

        messages = [
            msg
            for msg in session.message_history.messages
            if not (
                isinstance(msg, ModelRequest)
                and msg.metadata
                and msg.metadata.get("arox_internal")
            )
        ]
        history = build_state_history(messages)

        runtime = session.runner.runtime if session.runner is not None else None
        agent_config = self.config_loader.current_config.agent.get(session.agent_name)
        model = (
            runtime.provider_model
            if runtime is not None
            else session.extra.get("model_override")
            or (agent_config.model_ref if agent_config else None)
            or self.config_loader.current_config.model_ref
        )
        return {"history": history, "model": model}

    async def list_sessions(self):
        main_agent = self.config_loader.current_config.app.main_agent
        sessions = await self.session_manager.list_roots()
        views = []
        for stored in sessions:
            if not isinstance(stored, AgentSession) or stored.agent_name != main_agent:
                continue
            views.append(await build_session_view(stored))
        return views

    async def delete_session(self, session_id: str):
        session = await self._resolve_root_session(session_id)
        await self.session_manager.delete_tree(session)
        return {"status": "deleted"}

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="auto")
        server = uvicorn.Server(config)

        await server.serve()
