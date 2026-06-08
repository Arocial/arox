from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import override

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

from arox.core.app import app_setup, create_main_agent
from arox.core.chat import (
    ChatInputEvent,
    ChatInputReply,
    StepDoneEvent,
)
from arox.core.completion import parse_request
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
)
from arox.core.llm_base import LLMBaseAgent, MainAgent, ServerIdMapping
from arox.plugins.slots import SUBAGENTS

logger = logging.getLogger(__name__)

_user_prompt_metadata_patched = False


def _patch_vercel_user_prompt_metadata() -> None:
    """Carry a user turn's ``USER_INPUT_ID_KEY`` through ``dump_messages``.

    Stock ``_convert_user_prompt_part`` drops ``TextContent.metadata`` when it
    builds the ``TextUIPart``, so the fork anchor can't ride along. We wrap it
    and stash the id on the part's ``provider_metadata`` (the only
    round-trippable slot), under an ``arox`` wrapper key so it can't collide
    with pydantic_ai's own ``pydantic_ai`` wrapper. ``/state`` then hoists it to
    message-level ``metadata.custom``.
    """
    global _user_prompt_metadata_patched
    if _user_prompt_metadata_patched:
        return

    from pydantic_ai.messages import TextContent, UserPromptPart
    from pydantic_ai.ui.vercel_ai import _adapter
    from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessagePart

    from arox.core.session import USER_INPUT_ID_KEY

    original = _adapter._convert_user_prompt_part

    def _convert_with_metadata(part: UserPromptPart) -> list[UIMessagePart]:
        ui_parts = original(part)
        content = (
            part.content if isinstance(part.content, (list, tuple)) else [part.content]
        )
        # Match each dumped TextUIPart back to its source TextContent by text,
        # not by index: a CachePoint yields zero UI parts, so positions don't
        # line up. This assumes text -> id is unambiguous within one part, which
        # holds because arox builds exactly one TextContent per user input. The
        # assert fails loudly if that ever stops being true.
        ids_by_text: dict[str, str] = {}
        for item in content:
            if isinstance(item, TextContent) and isinstance(item.metadata, dict):
                input_id = item.metadata.get(USER_INPUT_ID_KEY)
                if input_id:
                    existing = ids_by_text.get(item.content)
                    assert existing is None or existing == str(input_id), (
                        "ambiguous user_input_id: two TextContents in one "
                        "UserPromptPart share text but carry different anchors"
                    )
                    ids_by_text[item.content] = str(input_id)
        for ui_part in ui_parts:
            if isinstance(ui_part, TextUIPart) and ui_part.text in ids_by_text:
                ui_part.provider_metadata = {
                    "arox": {USER_INPUT_ID_KEY: ids_by_text[ui_part.text]}
                }
        return ui_parts

    _adapter._convert_user_prompt_part = _convert_with_metadata  # ty: ignore[invalid-assignment]
    _user_prompt_metadata_patched = True


_patch_vercel_user_prompt_metadata()


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


class AgentInfo(BaseModel):
    id: str
    workspace: str
    main_agent: str
    subagents: list[str]


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


@dataclass
class AgentRun:
    main_agent: MainAgent
    task: asyncio.Task | None = None
    status: AgentTaskStatus = AgentTaskStatus.RUNNING
    error: Exception | None = None

    async def get_agent(self, name: str) -> LLMBaseAgent | None:
        if self.main_agent.name == name:
            return self.main_agent
        subagents = await self.main_agent.invoke_slot(SUBAGENTS)
        if subagents:
            for sa in subagents:
                if sa.name == name:
                    return sa
        return None

    async def get_subagent_names(self) -> list[str]:
        subagents = await self.main_agent.invoke_slot(SUBAGENTS)
        if subagents:
            return [sa.name for sa in subagents]
        return []


class VercelStreamIOAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.tool_ids = {}
        self.read_lock = asyncio.Lock()
        self.event_queues = {}
        self.pending_inputs: dict[IOEndpoint, ChatInputEvent] = {}
        self.run_instances: dict[str, AgentRun] = {}

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
                {"type": "cmd-input-request", "payload": event.generate_request()}
            )
            # Emit an explicit stream close signal to let the frontend decouple
            # stream management from business logic.
            messages.append({"type": "stream-close"})

        elif isinstance(event, StepDoneEvent):
            messages.append({"type": "step-done"})

        elif isinstance(event, ServerIdMapping):
            frame = {
                "type": "cmd-user-turn",
                "payload": {
                    "eventId": event.event_id,
                    "messageId": event.client_id,
                }
            }
            messages.append(frame)

        return messages

    def _format_event(self, adapter_io: IOEndpoint, event) -> list[str]:
        return [
            f"data: {json.dumps(m)}\n\n"
            for m in self._event_messages(adapter_io, event)
        ]

    async def _render_command_output(self, target, output: str | None) -> None:
        """Stream a command's text output through the normal event pipeline."""
        if not output:
            return
        await target.agent_io.send(output)

    async def _reissue_pending_input(self, target) -> None:
        """Re-emit the current pending ChatInputEvent so the client sees a
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
            req_id = reply.get("req_id")
            if not req_id:
                return {"status": "no_req_id"}
            deferred = reply.get("deferred_tools") or {}
            exception_input = reply.get("exception_input") or {}
            normal_input = reply.get("normal_input") or {}
            user_input = normal_input.get("user_input")

            # Intercept slash commands typed into the app so they don't
            # round-trip through the LLM. Mirrors the structured "command"
            # branch above and TextIOAdapter's slash handling.
            if isinstance(user_input, str) and user_input.startswith("/"):
                cmd_reply = await target.command_manager.try_handle_slash(user_input)
                if cmd_reply is not None:
                    await self._render_command_output(target, cmd_reply.output)
                    # The agent is still blocked on its current ChatInputEvent;
                    # re-emit it so the client can submit again.
                    await self._reissue_pending_input(target)
                    return {"status": "ok", "output": cmd_reply.output}

            client_message_id = reply.get("client_message_id")
            await target.adapter_io.send(
                ChatInputReply(
                    req_id=req_id,
                    deferred_answers=dict(deferred),
                    user_input=user_input,
                    retry=bool(exception_input.get("retry", False)),
                    client_message_id=client_message_id,
                )
            )
            self.pending_inputs.pop(target.adapter_io, None)
            return {"status": "ok"}

        return {"status": "noop"}

    async def ws_handler(self, websocket: WebSocket, target):
        from fastapi import WebSocketDisconnect

        adapter_io = target.adapter_io
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
                    await websocket.send_json({"type": "ack", "status": "ok"})
                else:
                    ack = await self._apply_input(target, payload)
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
        config_files: list[str | Path] | None = None,
        cli_args: list[str] | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        from contextlib import asynccontextmanager

        from arox.core.session import AgentSession, FileSessionStore, SessionManager

        self.config_files = config_files or []
        self.cli_args = cli_args or []
        self.host = host
        self.port = port
        self.parsed_config = app_setup(
            config_files=self.config_files, cli_args=self.cli_args
        )
        self.io_adapter = VercelStreamIOAdapter()
        self.session_store = FileSessionStore(
            max_age_days=self.parsed_config.app.session_max_age_days
        )
        self.session_manager = SessionManager(self.session_store)
        self.session_manager.register_session_type(AgentSession)

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
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.post("/api/agents", response_model=AgentInfo)(self.create_agent)
        self.app.get("/api/agents", response_model=list[AgentInfo])(self.list_agents)
        self.app.delete("/api/agents/{agent_id}")(self.delete_agent)
        self.app.websocket("/api/agents/{agent_id}/{agent_name}/ws")(self.ws)
        self.app.get(
            "/api/agents/{agent_id}/{agent_name}/suggestions",
            response_model=SuggestionResponse,
        )(self.suggestions)
        self.app.get("/api/agents/{agent_id}/{agent_name}/state")(self.state)
        self.app.get("/api/sessions", response_model=list[SessionInfo])(
            self.list_sessions
        )
        self.app.delete("/api/sessions/{session_id}")(self.delete_session)
        self.app.get("/api/health")(self.health)

    async def health(self):
        return {"status": "ok"}

    async def _get_agent(self, agent_id: str, agent_name: str):
        run_instance = self.io_adapter.run_instances.get(agent_id)
        if not run_instance:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        agent = await run_instance.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent {agent_name} not found."
            )
        return agent

    async def _get_agent_info(self, run_instance: AgentRun) -> AgentInfo:
        return AgentInfo(
            id=run_instance.main_agent.uuid,
            workspace=str(run_instance.main_agent.workspace),
            main_agent=run_instance.main_agent.name,
            subagents=await run_instance.get_subagent_names(),
        )

    async def create_agent(self, request: CreateAgentRequest):
        from arox.core.session import AgentSession

        parsed_config = self.parsed_config.model_copy(deep=True)

        if request.session_id:
            session = await self.session_store.load_session(request.session_id)
            if not session or not isinstance(session, AgentSession):
                raise HTTPException(
                    status_code=404, detail="Session not found or invalid"
                )
            parsed_config.app.main_agent = session.agent_name
            parsed_config.agent[session.agent_name] = session.agent_config
        else:
            session = AgentSession.create_initial(
                parsed_config, workspace=request.workspace
            )

        main_agent = create_main_agent(
            parsed_config,
            io_adapter=self.io_adapter,
            session=session,
            workspace=request.workspace,
        )
        main_agent.session.manager = self.session_manager

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

        return await self._get_agent_info(run_instance)

    async def list_agents(self):
        return [
            await self._get_agent_info(r)
            for r in self.io_adapter.run_instances.values()
        ]

    async def delete_agent(self, agent_id: str):
        run_instance = self.io_adapter.run_instances.pop(agent_id, None)
        if not run_instance:
            raise HTTPException(status_code=404, detail="Agent not found")
        if run_instance.task and not run_instance.task.done():
            run_instance.task.cancel()
        return {"status": "deleted"}

    async def ws(self, websocket: WebSocket, agent_id: str, agent_name: str):
        run_instance = self.io_adapter.run_instances.get(agent_id)
        if not run_instance:
            await websocket.close(code=4004, reason="agent not found")
            return
        agent = await run_instance.get_agent(agent_name)
        if not agent:
            await websocket.close(code=4004, reason="agent not found")
            return
        return await self.io_adapter.ws_handler(websocket, agent)

    async def suggestions(
        self,
        agent_id: str,
        agent_name: str,
        command: str | None = None,
        q: str | None = None,
    ):
        agent = await self._get_agent(agent_id, agent_name)
        return await self.io_adapter.suggestions(agent, command, q)

    async def state(self, agent_id: str, agent_name: str):
        agent = await self._get_agent(agent_id, agent_name)

        from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

        messages = agent.message_history
        ui_messages = VercelAIAdapter.dump_messages(messages)
        history = [
            msg.model_dump(mode="json", exclude_none=True) for msg in ui_messages
        ]

        # Each user message carries its USER_INPUT_ID_KEY on the dumped
        # TextUIPart's ``provider_metadata`` (injected by
        # ``_patch_vercel_user_prompt_metadata``). Hoist it to message-level
        # ``metadata.custom`` — the only slot @assistant-ui preserves for user
        # messages — so the client reads the fork anchor straight off the
        # message, the same shape it gets live via ``data-user-turn``. The id
        # travels with its own part, so no positional re-pairing is needed.
        # Strip the internal wrapper afterward to leave the wire payload as it
        # was before the patch.
        from arox.core.session import USER_INPUT_ID_KEY

        for msg in history:
            if msg.get("role") != "user":
                continue
            for part in msg.get("parts", []):
                provider_metadata = part.get("provider_metadata")
                if not isinstance(provider_metadata, dict):
                    continue
                arox_meta = provider_metadata.get("arox")
                if not isinstance(arox_meta, dict):
                    continue
                input_id = arox_meta.get(USER_INPUT_ID_KEY)
                if not input_id:
                    continue
                custom = msg.setdefault("metadata", {}).setdefault("custom", {})
                custom[USER_INPUT_ID_KEY] = input_id
                provider_metadata.pop("arox", None)
                if not provider_metadata:
                    part.pop("provider_metadata", None)
                break

        return {
            "history": history,
            "model": agent.provider_model,
        }

    async def list_sessions(self):
        from arox.core.session import AgentSession

        # The top-level session is the main agent's own AgentSession; filter to
        # the configured main agent so subagent sessions don't leak in.
        main_agent = self.parsed_config.app.main_agent

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
        await self.session_store.delete_session(session_id)
        return {"status": "deleted"}

    async def run(self):
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="auto")
        server = uvicorn.Server(config)

        await server.serve()
