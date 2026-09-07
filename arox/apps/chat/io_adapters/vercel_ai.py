from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override
from urllib.parse import parse_qs
from weakref import WeakValueDictionary

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, UIMessage
from starlette.types import ASGIApp, Receive, Scope, Send

from arox.core.agent_runtime import AgentRuntime
from arox.core.completion import parse_request
from arox.core.config import ConfigLoader
from arox.core.io import AbstractIOAdapter, IOEndpoint, SnapshotEvent
from arox.core.message_utils import visible_message_history
from arox.core.session import (
    MODEL_MESSAGE_ID_KEY,
    AgentSession,
    CommandCompletedEvent,
    ErrorEvent,
    SessionEvent,
    user_input_ids_on,
)
from arox.core.types import (
    USER_INPUT_ID_KEY,
    ClientInput,
    CommandPayload,
    MessagePayload,
    SessionTreeUpdate,
    TurnStateEvent,
)
from arox.plugins.compaction import CompactionEvent

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime
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
_WS_LOG_PAYLOAD_LIMIT = 1024


def _log_ws_payload(
    direction: Literal["IN", "OUT"], session_id: str, payload: object
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    size = len(serialized)
    if size > _WS_LOG_PAYLOAD_LIMIT:
        omitted = size - _WS_LOG_PAYLOAD_LIMIT
        serialized = (
            serialized[:_WS_LOG_PAYLOAD_LIMIT] + f"... <truncated {omitted} chars>"
        )
    message_type = (
        cast(dict[str, object], payload).get("type")
        if isinstance(payload, dict)
        else None
    )
    logger.debug(
        "WS %s session_id=%s type=%s size=%d payload=%s",
        direction,
        session_id,
        message_type,
        size,
        serialized,
    )


def _default_ui_message_id(
    message: ModelRequest | ModelResponse,
    role: Literal["system", "user", "assistant"],
    index: int,
) -> str:
    message_id = (message.metadata or {}).get(MODEL_MESSAGE_ID_KEY)
    if isinstance(message_id, str):
        return message_id
    return f"arox-{role}-{index}-{secrets.token_hex(8)}"


def dump_ui_messages(
    messages: Sequence[ModelMessage],
    *,
    generate_message_id: Callable[
        [ModelRequest | ModelResponse, Literal["system", "user", "assistant"], int],
        str,
    ]
    | None = None,
) -> list[dict]:
    """Convert model messages to serialized Vercel UI messages."""
    if generate_message_id is None:
        generate_message_id = _default_ui_message_id

    prepared_messages: list[ModelMessage] = []
    for message in messages:
        input_ids = user_input_ids_on(message)
        if not input_ids:
            prepared_messages.append(message)
            continue

        metadata = dict(message.metadata or {})
        custom = dict(metadata.get("custom") or {})
        custom[USER_INPUT_ID_KEY] = input_ids[-1]
        metadata["custom"] = custom
        prepared_messages.append(dataclasses.replace(message, metadata=metadata))

    ui_messages = VercelAIAdapter.dump_messages(
        prepared_messages, generate_message_id=generate_message_id
    )
    # `by_alias` to serialize keys as camel case, which assistant-ui
    # recognizes. See `pydantic_ai/ui/vercel_ai/_models.py:CamelBaseModel`
    return [
        msg.model_dump(mode="json", exclude_none=True, by_alias=True)
        for msg in ui_messages
    ]


def build_state_history(
    session: AgentSession,
    *,
    through_id: str | None = None,
) -> list[dict]:
    """Build the ordered history sent in a WebSocket state frame."""
    items: Sequence[ModelMessage | SessionEvent] = session.build_io_timeline(
        through_id=through_id
    )

    timeline: list[dict] = []
    message_batch: list[ModelMessage] = []

    def flush_messages() -> None:
        if not message_batch:
            return
        visible_messages = visible_message_history(message_batch)
        timeline.extend(
            {"type": "message", "message": message}
            for message in dump_ui_messages(visible_messages)
        )
        message_batch.clear()

    for item in items:
        if isinstance(item, CompactionEvent):
            flush_messages()
            timeline.append(
                {
                    "type": "compaction",
                    "event_id": item.id,
                    "trigger": item.trigger,
                    "llm_context_id": item.llm_context_id,
                    "timestamp": item.timestamp.isoformat(),
                }
            )
        elif isinstance(item, CommandCompletedEvent):
            flush_messages()
            payload = item.client_input.payload
            assert isinstance(payload, CommandPayload)
            timeline.append(
                {
                    "type": "command",
                    "client_message_id": item.client_input.client_message_id,
                    "server_message_id": item.client_input.server_message_id,
                    "command": payload.command,
                    "status": item.status,
                    "output": item.output,
                    "error": item.error,
                }
            )
        elif isinstance(item, (ModelRequest, ModelResponse)):
            message_batch.append(item)
    flush_messages()
    return timeline


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


@dataclasses.dataclass
class _SessionConnection:
    task: asyncio.Task
    websocket: WebSocket
    root_session: AgentSession
    target_session: AgentSession
    stream: VercelAIEventStream
    model: str | None
    runtime: AgentRuntime | None = None
    adapter_ep: IOEndpoint | None = None


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self, config_loader: ConfigLoader | None = None):
        super().__init__()
        self.config_loader = config_loader
        self._connections: dict[str, _SessionConnection] = {}
        self._event_contexts: dict[
            IOEndpoint,
            tuple[
                WebSocket,
                AgentSession,
                VercelAIEventStream,
                str | None,
                AgentSession,
            ],
        ] = {}
        self._connection_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )

    @override
    async def on_runtime_start(self, runtime: AgentRuntime) -> None:
        session_id = runtime.session.id
        connection_lock = self._connection_locks.setdefault(session_id, asyncio.Lock())
        async with connection_lock:
            connection = self._connections.get(session_id)
            if connection is not None:
                await self._attach_runtime(connection, runtime)

    @override
    async def on_runtime_stop(self, runtime: AgentRuntime) -> None:
        session_id = runtime.session.id
        connection_lock = self._connection_locks.setdefault(session_id, asyncio.Lock())
        async with connection_lock:
            connection = self._connections.get(session_id)
            if connection is not None:
                await self._detach_runtime(connection, runtime)

    async def _attach_runtime(
        self, connection: _SessionConnection, runtime: AgentRuntime
    ) -> None:
        if connection.runtime is runtime:
            return
        if connection.runtime is not None:
            await self._detach_runtime(connection, connection.runtime)
        adapter_ep = await self.connect(runtime)
        connection.runtime = runtime
        connection.adapter_ep = adapter_ep
        self._event_contexts[adapter_ep] = (
            connection.websocket,
            connection.root_session,
            connection.stream,
            connection.model,
            connection.target_session,
        )

    async def _detach_runtime(
        self, connection: _SessionConnection, runtime: AgentRuntime
    ) -> None:
        if connection.runtime is not runtime:
            return
        await self.disconnect(runtime)
        connection.runtime = None
        connection.adapter_ep = None

    @override
    async def handle_event(self, adapter_ep: IOEndpoint, event):
        context = self._event_contexts.get(adapter_ep)
        if context is None:
            return
        websocket, root_session, stream, model, target_session = context
        session_id = target_session.id
        if isinstance(event, SnapshotEvent):
            runtime = self.agent_io_for(adapter_ep)
            turn = getattr(runtime, "turn", None)
            await self._send_ws_json(
                websocket,
                {
                    "type": "state",
                    "history": build_state_history(
                        target_session, through_id=event.journal_id
                    )
                    if event.journal_id is not None
                    else [],
                    "model": model,
                    "busy": bool(turn and not turn.done),
                },
                session_id,
            )
            return

        for msg in await self._to_ui_messages(event, root_session, stream):
            await self._send_ws_json(websocket, msg, session_id)

    async def _send_ws_json(
        self, websocket: WebSocket, payload: dict, session_id: str
    ) -> None:
        _log_ws_payload("OUT", session_id, payload)
        await websocket.send_json(payload)

    @override
    async def on_endpoint_closed(self, adapter_ep: IOEndpoint) -> None:
        self._event_contexts.pop(adapter_ep, None)

    async def _to_ui_messages(
        self,
        event,
        root_session: AgentSession,
        stream: VercelAIEventStream,
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
            async for chunk in stream.handle_event(event):
                messages.append(chunk.model_dump(by_alias=True, exclude_none=True))

        elif isinstance(event, TurnStateEvent):
            messages.append({"type": "cmd-turn-state", "busy": event.busy})

        elif isinstance(event, ErrorEvent):
            messages.append({"type": "error", "errorText": event.error})

        elif isinstance(event, ClientInput):
            client_input = event
            payload = client_input.payload
            if isinstance(payload, MessagePayload) and payload.status == "started":
                input_content = payload.content
                if input_content is None:
                    return messages
                request = ModelRequest(parts=[UserPromptPart(content=input_content)])
                history = dump_ui_messages(
                    [request],
                    generate_message_id=lambda _msg, _role, _index: (
                        client_input.server_message_id
                    ),
                )
                if history:
                    message = history[0]
                    frame = {
                        "type": "cmd-client-input",
                        "client_message_id": client_input.client_message_id,
                        "server_message_id": client_input.server_message_id,
                        "payload": {
                            "type": "message",
                            "status": payload.status,
                            "message": message,
                        },
                    }
                    messages.append(frame)
            elif isinstance(payload, CommandPayload) and payload.status == "accepted":
                messages.append(
                    {
                        "type": "cmd-client-input",
                        "client_message_id": client_input.client_message_id,
                        "server_message_id": client_input.server_message_id,
                        "payload": {
                            "type": "command",
                            "status": payload.status,
                            "command": payload.command,
                        },
                    }
                )

        elif isinstance(event, CommandCompletedEvent):
            messages.append(
                {
                    "type": "cmd-command-completed",
                    "input": dataclasses.asdict(event.client_input),
                    "status": event.status,
                    "output": event.output,
                    "error": event.error,
                }
            )

        elif isinstance(event, SessionTreeUpdate):
            if event.session_id == root_session.id:
                messages.append(
                    {
                        "type": "cmd-session-tree",
                        **((await build_session_view(root_session)).model_dump()),
                    }
                )

        return messages

    async def _apply_input(self, target, adapter_ep: IOEndpoint, payload: dict) -> None:
        if payload.get("cancel"):
            runtime = target.session.runtime
            if runtime is not None:
                await runtime.cancel_turn()
            return None

        cmd = payload.get("command")
        if cmd is not None:
            await adapter_ep.send(
                ClientInput(
                    payload=CommandPayload(command=cmd),
                    client_message_id=payload.get("client_message_id"),
                )
            )
            return

        reply = payload.get("reply")
        if reply is not None:
            reply_message = UIMessage.model_validate(reply)
            if reply_message.metadata:
                custom_metadata = reply_message.metadata.get("custom", {})
            else:
                custom_metadata = {}
            reply_metadata = custom_metadata.get("chatInputEventResult")
            if not isinstance(reply_metadata, dict):
                raise ValueError("no chatInputEventResult")

            user_msg = VercelAIAdapter.load_messages([reply_message])
            parts = user_msg[0].parts
            assert len(parts) == 1
            from pydantic_ai.messages import UserPromptPart

            part = parts[0]
            assert isinstance(part, UserPromptPart)
            input_content = part.content
            chat_input = ClientInput(
                payload=MessagePayload(content=input_content),
                client_message_id=reply_metadata.get("client_message_id"),
            )

            await adapter_ep.send(chat_input)

    async def ws_handler(
        self,
        websocket: WebSocket,
        root_session: AgentSession,
        target_session: AgentSession,
        model: str | None,
    ):
        from fastapi import WebSocketDisconnect

        await websocket.accept()
        session_id = target_session.id
        connection_lock = self._connection_locks.setdefault(session_id, asyncio.Lock())
        connection_task = asyncio.current_task()
        assert connection_task is not None
        connection = _SessionConnection(
            task=connection_task,
            websocket=websocket,
            root_session=root_session,
            target_session=target_session,
            stream=VercelAIEventStream(run_input=SubmitMessage(id="", messages=[])),
            model=model,
        )
        previous_task: asyncio.Task | None = None
        async with connection_lock:
            previous = self._connections.get(session_id)
            self._connections[session_id] = connection
            if previous is not None and previous.task is not connection_task:
                await asyncio.gather(
                    previous.websocket.close(
                        code=4000, reason="Replaced by a newer connection."
                    ),
                    return_exceptions=True,
                )
                previous.task.cancel()
                previous_task = previous.task
            runtime = target_session.runtime
            if runtime is not None:
                await self._attach_runtime(connection, runtime)
            else:
                await self._send_ws_json(
                    websocket,
                    {
                        "type": "state",
                        "history": build_state_history(
                            target_session,
                        ),
                        "model": model,
                        "busy": False,
                    },
                    session_id,
                )
        if previous_task is not None:
            await asyncio.gather(previous_task, return_exceptions=True)

        async def pump_in():
            while True:
                payload = await websocket.receive_json()
                _log_ws_payload("IN", session_id, payload)

                if target_session.runtime is None and self.config_loader is not None:
                    await target_session.ensure_runtime(self.config_loader, self)

                async with connection_lock:
                    current = self._connections.get(session_id)
                    if current is connection:
                        current_runtime = target_session.runtime
                        adapter_ep = connection.adapter_ep
                        if (
                            current_runtime is None
                            or connection.runtime is not current_runtime
                            or adapter_ep is None
                        ):
                            continue
                        await self._apply_input(current_runtime, adapter_ep, payload)

        in_task = asyncio.create_task(pump_in())
        try:
            await in_task
        except WebSocketDisconnect:
            pass
        finally:
            in_task.cancel()
            await asyncio.gather(in_task, return_exceptions=True)
            async with connection_lock:
                current = self._connections.get(session_id)
                if current is connection:
                    self._connections.pop(session_id, None)
                    if connection.runtime is not None:
                        await self._detach_runtime(connection, connection.runtime)

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
        self.io_adapter = VercelStreamIOAdapter(config_loader)
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

    async def start_session(self, session_id: str) -> SessionView:
        session = await self._resolve_root_session(session_id)
        if session.is_active:
            raise HTTPException(status_code=409, detail="Session is already active.")
        await session.ensure_runtime(self.config_loader, self.io_adapter)
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
        return await self.io_adapter.ws_handler(
            websocket, root, target, self._model_for(target)
        )

    async def suggestions(
        self,
        root_session_id: str,
        target_session_id: str,
        command: str | None = None,
        q: str | None = None,
    ):
        root = await self._resolve_root_session(root_session_id)
        target = await self._resolve_child_session(root, target_session_id)
        runtime = await target.ensure_runtime(self.config_loader, self.io_adapter)
        return await self.io_adapter.suggestions(runtime, command, q)

    def _model_for(self, session: AgentSession) -> str | None:
        runtime = session.runtime
        agent_config = self.config_loader.current_config.agent.get(session.agent_name)
        return (
            runtime.provider_model
            if runtime is not None
            else session.extra.get("model_override")
            or (agent_config.model_ref if agent_config else None)
            or self.config_loader.current_config.model_ref
        )

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
