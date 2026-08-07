import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, override

from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import UIMessage

from arox.core.config import ConfigLoader

if TYPE_CHECKING:
    from arox.core.session import SessionManager


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
)
from starlette.types import ASGIApp, Receive, Scope, Send

from arox.core.chat import (
    ChatInputReply,
    ChatInputRequest,
    StepDoneEvent,
)
from arox.core.completion import parse_request
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)
from arox.core.llm_base import (
    AgentInfoUpdate,
    LLMBaseAgent,
    MainAgent,
    ServerIdMapping,
)
from arox.plugins.slots import SUBAGENTS


class TokenAuthASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        import os
        import secrets

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
            from urllib.parse import parse_qs

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
    import dataclasses
    import json

    from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
    from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

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


class CreateAgentRequest(BaseModel):
    workspace: str | None = None
    session_id: str | None = None


class SessionInfo(BaseModel):
    id: str
    main_agent: str
    created_at: str
    updated_at: str
    workspace: str | None
    metadata: dict


class AgentTaskStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    ERROR = "error"


class AgentInfo(BaseModel):
    id: str
    name: str
    status: str
    workspace: str | None = None
    subagents: list["AgentInfo"] = []


@dataclass
class AgentRun:
    main_agent: MainAgent
    task: asyncio.Task | None = None
    status: AgentTaskStatus = AgentTaskStatus.RUNNING
    error: Exception | None = None

    async def get_agent(self, uuid: str) -> LLMBaseAgent | None:
        if self.main_agent.uuid == uuid:
            return self.main_agent
        subagents = await self.main_agent.invoke_slot(SUBAGENTS)
        if subagents:
            for sa in subagents:
                if sa.uuid == uuid:
                    return sa
        return None

    async def _get_subagents(self) -> list["AgentInfo"]:
        subagents = await self.main_agent.invoke_slot(SUBAGENTS)
        if subagents:
            return [
                AgentInfo(
                    id=sa.uuid,
                    name=sa.name,
                    status=sa.status,
                    workspace=str(sa.workspace) if sa.workspace else None,
                    subagents=[],
                )
                for sa in subagents
            ]
        return []

    async def get_agent_info(self) -> AgentInfo:
        return AgentInfo(
            id=self.main_agent.uuid,
            name=self.main_agent.name,
            status=self.main_agent.status,
            workspace=str(self.main_agent.workspace)
            if self.main_agent.workspace
            else None,
            subagents=await self._get_subagents(),
        )


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.read_lock = asyncio.Lock()
        self.event_queues = {}
        self.pending_inputs: dict[IOEndpoint, ChatInputRequest] = {}
        self.run_instances: dict[str, AgentRun] = {}
        self.event_streams = {}

    @override
    async def handle_event(self, adapter_io: IOEndpoint, event):
        if isinstance(event, ChatInputRequest):
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

    async def _event_messages(
        self, adapter_io: IOEndpoint, event, target_run: AgentRun, agent: LLMBaseAgent
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
            from pydantic_ai.ui.vercel_ai import VercelAIEventStream
            from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage

            stream = self.event_streams.setdefault(
                adapter_io,
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

        else:
            if isinstance(event, AgentInfoUpdate):
                if event.agent_uuid == target_run.main_agent.uuid:
                    messages.append(
                        {
                            "type": "cmd-agent-info",
                            **((await target_run.get_agent_info()).model_dump()),
                        }
                    )

        return messages

    async def _render_command_output(self, target, output: str | None) -> None:
        """Stream a command's text output through the normal event pipeline."""
        if not output:
            return
        await target.agent_io.send(output)

    async def _reissue_pending_input(self, target) -> None:
        """Re-emit the current pending ChatInputRequest so the client sees a
        fresh ``data-input-request``. Used after handling a command that did
        not consume the agent's pending input."""
        event = self.pending_inputs.get(target.adapter_io)
        if event is None:
            return
        queue = self.event_queues.setdefault(target.adapter_io, asyncio.Queue())
        await queue.put((target.adapter_io, event))

    async def _apply_input(self, target, payload: dict) -> dict:
        if payload.get("cancel"):
            from arox.core.llm_base import LLMBaseAgent

            if isinstance(target, LLMBaseAgent):
                target.cancel_foreground_task()
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

            text_input = chat_input_reply.text_content

            # Intercept slash commands typed into the app so they don't
            # round-trip through the LLM. Mirrors the structured "command"
            # branch above and TextIOAdapter's slash handling.
            if text_input and text_input.startswith("/"):
                cmd_reply = await target.command_manager.try_handle_slash(text_input)
                if cmd_reply is not None:
                    await self._render_command_output(target, cmd_reply.output)
                    # The agent is still blocked on its current ChatInputRequest;
                    # re-emit it so the client can submit again.
                    await self._reissue_pending_input(target)
                    return {"status": "ok", "output": cmd_reply.output}

            await target.adapter_io.send(chat_input_reply)
            self.pending_inputs.pop(target.adapter_io, None)
            return {"status": "ok"}

        return {"status": "noop"}

    async def ws_handler(
        self, websocket: WebSocket, target_run: AgentRun, agent: LLMBaseAgent
    ):
        from fastapi import WebSocketDisconnect

        adapter_io = agent.adapter_io
        queue = self.event_queues.setdefault(adapter_io, asyncio.Queue())

        await websocket.accept()

        async def pump_out():
            while True:
                _io, event = await queue.get()
                for msg in await self._event_messages(
                    adapter_io, event, target_run, agent
                ):
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
                    event = self.pending_inputs.get(adapter_io)
                    if event is not None:
                        for msg in await self._event_messages(
                            adapter_io, event, target_run, agent
                        ):
                            await websocket.send_json(msg)
                    await websocket.send_json({"type": "ack", "status": "ok"})
                else:
                    ack = await self._apply_input(agent, payload)
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
        req = parse_request(text, agent=target)
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
        self.session_store = session_manager.session_store

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with self.session_manager, self.io_adapter:
                yield
            # Cancel all running app tasks on shutdown
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
        self.app.add_middleware(TokenAuthASGIMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.get("/api/agents", response_model=list[AgentInfo])(self.list_agents)
        self.app.post("/api/agents", response_model=AgentInfo)(self.create_agent)
        self.app.delete("/api/agents/{main_agent_uuid}")(self.delete_agent)
        self.app.websocket("/api/agents/{main_agent_uuid}/{subagent_uuid}/ws")(self.ws)
        self.app.get(
            "/api/agents/{main_agent_uuid}/{subagent_uuid}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)
        self.app.get("/api/agents/{main_agent_uuid}/{subagent_uuid}/state")(self.state)
        self.app.get("/api/sessions", response_model=list[SessionInfo])(
            self.list_sessions
        )
        self.app.delete("/api/sessions/{session_id}")(self.delete_session)
        self.app.get("/api/health")(self.health)

    async def health(self):
        return {"status": "ok"}

    async def _get_agent(self, main_agent_uuid: str, subagent_uuid: str):
        run_instance = self.io_adapter.run_instances.get(main_agent_uuid)
        if not run_instance:
            raise HTTPException(
                status_code=404, detail=f"Agent {main_agent_uuid} not found."
            )
        agent = await run_instance.get_agent(subagent_uuid)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent {subagent_uuid} not found."
            )
        return agent

    async def list_agents(self) -> list[AgentInfo]:
        agents = []
        for run_instance in self.io_adapter.run_instances.values():
            agents.append(await run_instance.get_agent_info())
        return agents

    async def create_agent(self, request: CreateAgentRequest):
        from arox.core.session import AgentSession

        session = None
        if request.session_id:
            session = await self.session_store.load_session([request.session_id])
            if not session or not isinstance(session, AgentSession):
                raise HTTPException(
                    status_code=404, detail="Session not found or invalid"
                )

        if session is None:
            main_agent_name = self.config_loader.current_config.app.main_agent
            agent_config = self.config_loader.current_config.agent.get(main_agent_name)
            agent_type = agent_config.type if agent_config else "chat"
            session = AgentSession(
                agent_name=main_agent_name,
                agent_type=agent_type,
                workspace=str(Path(request.workspace).absolute())
                if request.workspace
                else None,
                manager=self.session_manager,
            )
        else:
            session.manager = self.session_manager
            if request.workspace:
                session.workspace = str(Path(request.workspace).absolute())

        main_agent = session.create_agent(
            config_loader=self.config_loader,
            io_adapter=self.io_adapter,
        )
        if not isinstance(main_agent, MainAgent):
            raise TypeError(f"Main agent '{session.agent_name}' must be a MainAgent")

        run_instance = AgentRun(main_agent=main_agent)
        self.io_adapter.run_instances[main_agent.uuid] = run_instance

        async def run_agent():
            async with main_agent:
                if request.session_id:
                    await main_agent.agent_io.send(
                        f"Session restored: {request.session_id}"
                    )
                await main_agent.run()

        task = asyncio.create_task(run_agent())
        run_instance.task = task

        def on_task_done(t: asyncio.Task):
            try:
                t.result()
                run_instance.status = AgentTaskStatus.STOPPED
                logger.info(f"Agent {main_agent.uuid} finished normally.")
            except asyncio.CancelledError:
                run_instance.status = AgentTaskStatus.CANCELLED
                logger.info(f"Agent {main_agent.uuid} was cancelled.")
            except Exception as e:
                run_instance.status = AgentTaskStatus.ERROR
                run_instance.error = e
                logger.exception(f"Agent {main_agent.uuid} crashed with error.")

        task.add_done_callback(on_task_done)

        return await run_instance.get_agent_info()

    async def delete_agent(self, main_agent_uuid: str):
        run_instance = self.io_adapter.run_instances.pop(main_agent_uuid, None)
        if not run_instance:
            raise HTTPException(status_code=404, detail="Agent not found")
        if run_instance.task and not run_instance.task.done():
            run_instance.task.cancel()
        return {"status": "deleted"}

    async def ws(self, websocket: WebSocket, main_agent_uuid: str, subagent_uuid: str):
        run_instance = self.io_adapter.run_instances.get(main_agent_uuid)
        if not run_instance:
            await websocket.close(code=4004, reason="agent not found")
            return
        agent = await run_instance.get_agent(subagent_uuid)
        if not agent:
            await websocket.close(code=4004, reason="agent not found")
            return
        return await self.io_adapter.ws_handler(websocket, run_instance, agent)

    async def suggestions(
        self,
        main_agent_uuid: str,
        subagent_uuid: str,
        command: str | None = None,
        q: str | None = None,
    ):
        agent = await self._get_agent(main_agent_uuid, subagent_uuid)
        return await self.io_adapter.suggestions(agent, command, q)

    async def state(self, main_agent_uuid: str, subagent_uuid: str):
        agent = await self._get_agent(main_agent_uuid, subagent_uuid)

        from pydantic_ai import ModelRequest

        messages = [
            msg
            for msg in agent.message_history
            if not (
                isinstance(msg, ModelRequest)
                and msg.metadata
                and msg.metadata.get("arox_internal")
            )
        ]
        history = build_state_history(messages)

        return {
            "history": history,
            "model": agent.provider_model,
        }

    async def list_sessions(self):
        from arox.core.session import AgentSession

        # The top-level session is the main agent's own AgentSession; filter to
        # the configured main agent so subagent sessions don't leak in.
        main_agent = self.config_loader.current_config.app.main_agent

        sessions = await self.session_store.list_sessions()
        return [
            SessionInfo(
                id=s.id,
                main_agent=s.agent_name,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
                workspace=s.workspace,
                metadata=s.metadata,
            )
            for s in sessions
            if isinstance(s, AgentSession) and s.agent_name == main_agent
        ]

    async def delete_session(self, session_id: str):
        await self.session_store.delete_session([session_id])
        return {"status": "deleted"}

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="auto")
        server = uvicorn.Server(config)

        await server.serve()
