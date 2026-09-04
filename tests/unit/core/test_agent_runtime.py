import asyncio
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.usage import RunUsage

from arox.core.agent_runtime import AgentDeps, AgentRuntime, ContinueAgentRun
from arox.core.app import app_setup
from arox.core.background import BackgroundTaskBroker
from arox.core.io import AbstractIOAdapter, IOEndpoint, SnapshotEvent
from arox.core.message_utils import (
    AROX_INTERNAL_KEY,
    internal_user_prompt_part,
    visible_message_history,
)
from arox.core.plugin import (
    CommandDispatchResult,
    CommandReply,
    Plugin,
    tool,
)
from arox.core.session import (
    AgentSession,
    CommandCompletedEvent,
    ModelMessageEvent,
)
from arox.core.types import (
    ClientInput,
    CommandPayload,
    MessagePayload,
    TurnStateEvent,
    normalize_client_input,
)
from arox.plugins.core import SetModelEvent
from tests.history import compact_history, contains_input, context_resets, reset_history


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_ep, event):
        pass


class _FailingToolPlugin(Plugin):
    @tool()
    def fail(self) -> None:
        raise RuntimeError("expected failure")


@pytest.mark.asyncio
async def test_turn_input_waits_for_history_lock():
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.history_lock = asyncio.Lock()
    runtime._record_user_input = AsyncMock()
    await runtime.history_lock.acquire()

    task = asyncio.create_task(
        runtime._run_turn_input(ClientInput(payload=MessagePayload(content="hello")))
    )
    await asyncio.sleep(0)

    runtime._record_user_input.assert_not_awaited()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    runtime.history_lock.release()


@pytest.mark.asyncio
async def test_command_dispatch_records_request_and_completion_timeline():
    session = AgentSession(agent_name="main")

    async def dispatch(command):
        assert command == "/info"
        return CommandDispatchResult(
            "handled",
            CommandReply(output="details"),
        )

    class Endpoint:
        def __init__(self):
            self.sent = []

        async def send(self, event):
            self.sent.append(event)

    endpoint = Endpoint()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            session=session,
            command_manager=SimpleNamespace(dispatch=dispatch),
            agent_ep=endpoint,
        ),
    )

    client_input = ClientInput(
        payload=CommandPayload(command="/info"),
        server_message_id="command-id",
    )
    result = await AgentRuntime._dispatch_command(runtime, client_input)

    assert result.status == "handled"
    assert len(endpoint.sent) == 1
    assert isinstance(endpoint.sent[0], CommandCompletedEvent)
    assert endpoint.sent[0].output == "details"
    assert len(session.journal) == 1
    completed = session.journal[0]
    assert isinstance(completed, CommandCompletedEvent)
    assert completed.client_input is client_input
    assert completed.output == "details"


@pytest.mark.asyncio
async def test_accept_command_emits_accepted_client_input(monkeypatch):
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.session = AgentSession(path=["command-session"], agent_name="main")
    runtime._command_tasks = set()
    runtime.agent_ep = SimpleNamespace(send=AsyncMock())
    run_command = AsyncMock()
    monkeypatch.setattr(runtime, "_run_command", run_command)
    client_input = ClientInput(
        payload=CommandPayload(command="/info"),
        client_message_id="client-command-1",
    )

    accepted = await runtime.accept_input(client_input)
    await asyncio.gather(*runtime._command_tasks)

    assert accepted.payload.status == "accepted"
    assert accepted.server_message_id
    runtime.agent_ep.send.assert_awaited_once_with(accepted)
    run_command.assert_awaited_once_with(accepted)


@pytest.mark.asyncio
async def test_llm_notifications_are_enqueued_once_after_node_run():
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.background_tasks = BackgroundTaskBroker()
    runtime._pending_user_inputs = deque()
    runtime.notify_llm("First task finished.")
    runtime.notify_llm("Second task finished.")
    runtime._journal_history_initialized = True
    ctx = SimpleNamespace(enqueue=Mock(), run_id=None)
    result = object()

    returned = await runtime._wrap_node_run(
        ctx, node=object(), handler=AsyncMock(return_value=result)
    )

    assert returned is result
    ctx.enqueue.assert_called_once_with(
        "First task finished.\n\nSecond task finished.", priority="asap"
    )
    assert not runtime.background_tasks.drain_notices()

    await runtime._wrap_node_run(
        ctx, node=object(), handler=AsyncMock(return_value=result)
    )
    ctx.enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_pending_user_inputs_are_enqueued_after_node_run():
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.background_tasks = BackgroundTaskBroker()
    pending_payload = MessagePayload(content="steer")
    pending_input = ClientInput(payload=pending_payload)
    runtime._pending_user_inputs = deque([pending_input])
    runtime._record_user_input = AsyncMock()
    runtime._journal_history_initialized = True
    ctx = SimpleNamespace(enqueue=Mock(), run_id=None)

    result = object()
    returned = await runtime._wrap_node_run(
        ctx, node=object(), handler=AsyncMock(return_value=result)
    )

    assert returned is result
    runtime._record_user_input.assert_awaited_once_with(pending_input)
    content = pending_payload.content
    assert content is not None
    ctx.enqueue.assert_called_once_with(*content, priority="asap")
    assert not runtime._pending_user_inputs


@pytest.mark.asyncio
async def test_run_error_logs_exception_traceback(caplog):
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(session=AgentSession(agent_name="test"), new_message_index=0),
    )
    ctx = cast(
        RunContext[AgentDeps],
        SimpleNamespace(
            usage=RunUsage(),
            messages=[],
            run_id="run-id",
            conversation_id="conversation-id",
            metadata=None,
        ),
    )

    try:
        raise RuntimeError("model failed")
    except RuntimeError as error:
        raised_error = error
        with caplog.at_level("ERROR", logger="arox.core.agent_runtime"):
            result = await AgentRuntime._on_run_error(runtime, ctx, error=error)

    assert result.output is raised_error
    assert "Agent run failed." in caplog.text
    assert "RuntimeError: model failed" in caplog.text
    assert "test_run_error_logs_exception_traceback" in caplog.text


@asynccontextmanager
async def _managed_runtime(runtime, config_loader, io_adapter):
    async with runtime:
        yield runtime


def test_internal_binary_request_is_hidden_without_vendor_metadata():
    binary = BinaryContent(data=b"contents", media_type="application/octet-stream")
    internal = ModelRequest(parts=[internal_user_prompt_part([binary])])
    marked_content = internal.parts[0].content
    assert isinstance(marked_content, list)
    assert isinstance(marked_content[0], TextContent)
    assert marked_content[0].metadata == {AROX_INTERNAL_KEY: True}
    assert marked_content[1] is binary
    assert binary.vendor_metadata is None

    merged = ModelRequest(
        parts=[*internal.parts, UserPromptPart(content="visible question")]
    )
    visible = visible_message_history([merged])
    assert len(visible) == 1
    assert isinstance(visible[0], ModelRequest)
    assert len(visible[0].parts) == 1
    visible_part = visible[0].parts[0]
    assert isinstance(visible_part, UserPromptPart)
    assert visible_part.content == "visible question"


@pytest.mark.asyncio
async def test_internal_request_stays_hidden_after_next_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    session = AgentSession(path=["internal-history"], agent_name="test_agent")
    reset_history(
        session,
        [
            ModelRequest(parts=[internal_user_prompt_part("<file>secret</file>")]),
            ModelRequest(parts=[UserPromptPart(content="first question")]),
            ModelResponse(parts=[TextPart(content="first answer")]),
        ],
    )
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    runtime = AgentRuntime(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=session,
    )

    async def stream_function(messages, info):
        yield "done"

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            await runtime.run_turn("second question")

    first_request = session.message_history[0]
    assert isinstance(first_request, ModelRequest)
    assert not first_request.metadata
    internal_content = first_request.parts[0].content
    assert isinstance(internal_content, list)
    assert isinstance(internal_content[0], TextContent)
    assert internal_content[0].metadata == {AROX_INTERNAL_KEY: True}

    visible = visible_message_history(session.message_history)
    visible_text_parts = []
    for message in visible:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            if isinstance(part.content, str):
                visible_text_parts.append(part.content)
            else:
                visible_text_parts.extend(
                    item.content
                    for item in part.content
                    if isinstance(item, TextContent)
                )
    visible_text = "\n".join(visible_text_parts)
    assert "<file>secret</file>" not in visible_text
    assert "first question" in visible_text
    assert "second question" in visible_text


@pytest.mark.asyncio
async def test_agent_skills_filtering(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = ["skill1"]
""")

    # Monkeypatch Path.cwd to return tmp_path so ConfigLoader finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        assert "skill1" in runtime.skill_catalog
    assert "skill2" not in runtime.skill_catalog


@pytest.mark.asyncio
async def test_agent_skills_string(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
skills = "skill2"
""")

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        assert "skill1" not in runtime.skill_catalog
    assert "skill2" in runtime.skill_catalog


@pytest.mark.asyncio
async def test_agent_skills_none(tmp_path, monkeypatch):
    # Create dummy skills
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    skill1_dir = skills_dir / "skill1"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: Skill 1
---
""")

    skill2_dir = skills_dir / "skill2"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Skill 2
---
""")

    # Create dummy config
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi there."
""")

    # Monkeypatch Path.cwd to return tmp_path so discover_skills finds the skills
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    io_adapter = _StubIOAdapter()

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=AgentSession(
            path=["dummy"],
            agent_name="test_agent",
        ),
    )

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        assert "skill1" in runtime.skill_catalog
    assert "skill2" in runtime.skill_catalog


@pytest.mark.asyncio
async def test_event_dispatches_to_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    class CustomEvent:
        pass

    received: list[CustomEvent] = []

    runtime = AgentRuntime(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )

    async def handler(event):
        received.append(event)

    runtime.agent_ep.register_event_handler(CustomEvent, handler)

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        ev = CustomEvent()
        endpoint = next(iter(runtime.io_adapter.adapter_ep_to_runtime))
        await endpoint.send(ev)

    assert received == [ev]


@pytest.mark.asyncio
async def test_set_model_event_updates_model_ref(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(
        cli_args={"workspace": str(tmp_path)},
    )

    runtime = AgentRuntime(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )

    calls: list[str] = []
    original_set_model = runtime.set_model

    def spy(ref):
        calls.append(ref)
        original_set_model(ref)

    runtime.set_model = spy  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        calls.clear()
        runtime.agent_ep.register_event_handler(
            SetModelEvent, lambda e: runtime.set_model(e.model_ref)
        )
        endpoint = next(iter(runtime.io_adapter.adapter_ep_to_runtime))
        await endpoint.send(SetModelEvent(model_ref="test"))

    assert calls == ["test"]
    assert runtime.model_ref == "test"


def test_build_skill_catalog():
    assert AgentRuntime._build_skill_catalog({}) == ""

    skills = {
        "test_skill": {
            "name": "test_skill",
            "description": "A test skill",
            "location": "/path/to/SKILL.md",
        }
    }

    catalog = AgentRuntime._build_skill_catalog(skills)
    assert "<available_skills>" in catalog
    assert "<name>test_skill</name>" in catalog
    assert "<description>A test skill</description>" in catalog
    assert "<location>/path/to/SKILL.md</location>" in catalog


@pytest.mark.asyncio
async def test_inference_continues_with_replacement_context():
    replacement: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="summary")])
    ]
    restarted = SimpleNamespace(
        output=ContinueAgentRun(replacement),
        new_messages=lambda: [],
        all_messages=lambda: replacement,
    )
    completed = SimpleNamespace(
        output="done",
        new_messages=lambda: [],
        all_messages=lambda: replacement,
    )
    run_mock = AsyncMock(side_effect=[restarted, completed])
    session = AgentSession(agent_name="test")
    reset_history(session, replacement)
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            model_ref="test",
            model=object(),
            provider_model="test",
            model_params={},
            request_limit=None,
            request_limit_prompt="",
            run_info=SimpleNamespace(run_id=None),
            session=session,
            agent_ep=SimpleNamespace(),
            _pydantic_agent=SimpleNamespace(run=run_mock),
            _handle_stream_output=AsyncMock(),
            set_model=Mock(),
        ),
    )

    result = await AgentRuntime._run_inference(
        runtime,
        "question",
        message_history=[],
    )

    assert result is completed
    calls = run_mock.await_args_list
    assert calls[0].args == ("question",)
    assert calls[1].args == (None,)
    assert calls[1].kwargs["message_history"] is replacement
    assert session.message_history == replacement


@pytest.mark.asyncio
async def test_request_limit_prompt_continues_with_native_usage_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
request_limit = 1
request_limit_prompt = "Check your progress and continue."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    runtime = AgentRuntime(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["dummy"], agent_name="test_agent"),
    )
    tool_executions = 0

    def ping():
        nonlocal tool_executions
        tool_executions += 1
        return "pong"

    runtime.add_local_tool(ping)
    requests = []

    async def stream_function(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            yield {0: DeltaToolCall(name="ping", json_args="{}")}
        else:
            yield "done"

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            result = await runtime.run_turn("start")

    assert result.output == "done"
    assert len(requests) == 2
    assert tool_executions == 1
    parts = [part for message in result.all_messages() for part in message.parts]
    assert any(isinstance(part, ToolCallPart) for part in parts)
    assert any(
        isinstance(part, ToolReturnPart) and part.content == "pong" for part in parts
    )
    user_prompts = [part.content for part in parts if isinstance(part, UserPromptPart)]
    assert not isinstance(user_prompts[0], str)
    assert isinstance(user_prompts[0][0], TextContent)
    assert user_prompts[0][0].content == "start"
    assert user_prompts[1] == "Check your progress and continue."


@pytest.mark.asyncio
async def test_plugin_tool_error_is_returned_without_ending_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(
        "arox.utils.import_class", lambda *_args, **_kwargs: _FailingToolPlugin
    )
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
plugins = ["failing"]
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    runtime = AgentRuntime(
        config_loader,
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["tool-error"], agent_name="test_agent"),
    )
    requests = []

    async def stream_function(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            yield {0: DeltaToolCall(name="fail", json_args="{}")}
        else:
            yield "recovered"

    async with _managed_runtime(runtime, config_loader, runtime.io_adapter):
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            result = await runtime.run_turn("start")

    assert result.output == "recovered"
    assert len(requests) == 2
    tool_return = next(
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    assert tool_return.content == {
        "ok": False,
        "error": {
            "type": "RuntimeError",
            "message": "expected failure",
            "retryable": False,
        },
    }


@pytest.mark.asyncio
async def test_inference_error_is_recorded_in_session_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}),
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["error-snapshot"], agent_name="test_agent"),
    )
    error_result = SimpleNamespace(
        output=RuntimeError("model failed"),
        all_messages=lambda: [],
        new_messages=lambda: [],
    )

    async def fail_agent_run(*args, **kwargs):
        return error_result

    monkeypatch.setattr(runtime._pydantic_agent, "run", fail_agent_run)
    sent_events = []
    original_send = runtime.agent_ep.send

    async def capture_event(event):
        sent_events.append(event)
        return await original_send(event)

    monkeypatch.setattr(runtime.agent_ep, "send", capture_event)

    async with _managed_runtime(runtime, runtime.config_loader, runtime.io_adapter):
        with pytest.raises(RuntimeError, match="model failed"):
            await runtime.run_turn("fail")

    user_message = next(
        event
        for event in sent_events
        if isinstance(event, ClientInput)
        and isinstance(event.payload, MessagePayload)
        and event.payload.status == "started"
    )
    assert user_message.payload.text_content == "fail"
    session_snapshot = runtime.session.build_io_timeline()
    assert len(session_snapshot) == 2
    assert isinstance(session_snapshot[0], ModelRequest)
    assert isinstance(session_snapshot[1], ModelResponse)
    assert session_snapshot[1].text == "RuntimeError: model failed"

    assert runtime.agent_ep._safe_journal_id == runtime.session.journal[-1].id


@pytest.mark.asyncio
async def test_inference_cancellation_sends_only_formatted_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}),
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["cancelled-inference"], agent_name="test_agent"),
    )
    cancelled_result = SimpleNamespace(
        output=asyncio.CancelledError(),
        all_messages=lambda: [],
        new_messages=lambda: [],
    )

    async def cancel_agent_run(*args, **kwargs):
        return cancelled_result

    monkeypatch.setattr(runtime._pydantic_agent, "run", cancel_agent_run)
    sent_events = []
    original_send = runtime.agent_ep.send

    async def capture_event(event):
        sent_events.append(event)
        return await original_send(event)

    monkeypatch.setattr(runtime.agent_ep, "send", capture_event)

    async with _managed_runtime(runtime, runtime.config_loader, runtime.io_adapter):
        with pytest.raises(asyncio.CancelledError):
            await runtime.run_turn("cancel")

    assert sent_events[-2] == "Task interrupted."
    assert sent_events[-1] == TurnStateEvent(busy=False)
    assert all(event.event_type != "error" for event in runtime.session.journal)


@pytest.mark.asyncio
async def test_agent_lifecycle_session_binding_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-id"], agent_name="test_agent")

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    assert runtime.uuid == session.id == "test-session-id"
    assert session.runtime is None
    assert session.is_active is False
    assert not hasattr(runtime, "status")
    assert runtime not in io_adapter.adapter_ep_to_runtime.values()

    async with _managed_runtime(runtime, config_loader, io_adapter):
        assert session.runtime is runtime
        assert session.is_active is True
        endpoint = next(iter(io_adapter.adapter_ep_to_runtime))
        assert io_adapter.agent_io_for(endpoint) is runtime

    assert session.runtime is None
    assert session.is_active is False
    assert runtime not in io_adapter.adapter_ep_to_runtime.values()


@pytest.mark.asyncio
async def test_agent_manages_current_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}),
        io_adapter=_StubIOAdapter(),
        session=AgentSession(path=["managed-task"], agent_name="test_agent"),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    consumed_inputs = []

    async def blocking_turn(client_input):
        consumed_inputs.append(client_input.payload.text_content)
        started.set()
        await release.wait()
        return SimpleNamespace(output=client_input)

    runtime._run_turn_input = blocking_turn  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    async with runtime:
        turn = runtime.start_message("work")
        assert turn is not None
        task = turn.task
        await started.wait()
        assert runtime.turn is turn
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.01)
        parallel_turn = runtime.start_message("parallel work")
        assert parallel_turn is turn

        assert await runtime.cancel_turn()
        assert task.cancelled()
        assert not await runtime.cancel_turn()
        assert turn.result is None
        assert isinstance(turn.error, asyncio.CancelledError)
        assert consumed_inputs == ["work"]

        release.set()
        completed_turn = runtime.start_message("completed work")
        assert completed_turn is not None
        completed_result = await completed_turn
        assert isinstance(completed_result.output, ClientInput)
        assert completed_result.output.payload.text_content == "completed work"
        assert runtime.turn is completed_turn
        assert completed_turn.result is completed_result
        assert completed_turn.error is None

        started.clear()
        release.clear()
        consumed_inputs.clear()
        queued_turn = runtime.start_message("first queued work")
        assert queued_turn is not None
        await started.wait()
        assert runtime.start_message("second queued work") is queued_turn
        runtime._journal_history_initialized = True
        run_context = SimpleNamespace(enqueue=Mock(), run_id=None)
        result = object()
        returned = await runtime._wrap_node_run(
            cast(RunContext[AgentDeps], run_context),
            node=object(),
            handler=AsyncMock(return_value=result),
        )
        assert returned is result
        enqueued_content = run_context.enqueue.call_args.args
        assert isinstance(enqueued_content[0], TextContent)
        assert enqueued_content[0].content == "second queued work"
        assert not runtime._pending_user_inputs
        release.set()
        queued_result = await queued_turn
        assert consumed_inputs == ["first queued work"]
        assert queued_result.output.payload.text_content == "first queued work"

        started.clear()
        release.clear()
        consumed_inputs.clear()
        trailing_turn = runtime.start_message("last model request")
        assert trailing_turn is not None
        await started.wait()
        assert runtime.start_message("missed injection window") is trailing_turn
        release.set()
        trailing_result = await trailing_turn
        assert consumed_inputs == ["last model request", "missed injection window"]
        assert trailing_result.output.payload.text_content == "missed injection window"

        async def failed_turn(user_input):
            raise RuntimeError("model failed")

        runtime._run_turn_input = failed_turn  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
        with pytest.raises(RuntimeError, match="model failed"):
            failed_turn_handle = runtime.start_message("failed work")
            assert failed_turn_handle is not None
            await failed_turn_handle
        assert failed_turn_handle.result is None
        assert isinstance(failed_turn_handle.error, RuntimeError)


@pytest.mark.asyncio
async def test_agent_lifecycle_exception_sets_error_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-err"], agent_name="test_agent")

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    with pytest.raises(RuntimeError, match="something broke"):
        async with _managed_runtime(runtime, config_loader, io_adapter):
            raise RuntimeError("something broke")

    assert session.runtime is None
    assert session.journal[-1].event_type == "error"
    assert "RuntimeError: something broke" in session.journal[-1].error
    assert runtime not in io_adapter.adapter_ep_to_runtime.values()


@pytest.mark.asyncio
async def test_agent_lifecycle_cancellation_sets_interrupted_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(path=["test-session-cancel"], agent_name="test_agent")

    runtime = AgentRuntime(
        config_loader,
        io_adapter=io_adapter,
        session=session,
    )

    with pytest.raises(asyncio.CancelledError):
        async with _managed_runtime(runtime, config_loader, io_adapter):
            raise asyncio.CancelledError()

    assert session.runtime is None
    assert session.journal[-1].event_type == "error"
    assert session.journal[-1].error == "Task interrupted."
    assert runtime not in io_adapter.adapter_ep_to_runtime.values()


@pytest.mark.asyncio
async def test_runtime_uses_session_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("""
model_ref = "test"
[agent.test_agent]
system_prompt = "Hi."
""")
    config_loader = app_setup(cli_args={"workspace": str(tmp_path)})
    io_adapter = _StubIOAdapter()
    session = AgentSession(
        path=["parent-id", "child-session-id"],
        agent_name="test_agent",
    )

    runtime = AgentRuntime(config_loader, io_adapter, session)
    async with runtime:
        assert runtime.uuid == "child-session-id"
        assert runtime.session is session
        assert runtime.name == "test_agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_tools", [False, True])
async def test_steering_messages_are_journaled_before_later_fork_boundaries(
    tmp_path, monkeypatch, use_tools
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "test"\n[agent.main]\nsystem_prompt = "Hi."\n')
    session = AgentSession(agent_name="main")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}), _StubIOAdapter(), session
    )

    def ping():
        return "pong"

    runtime.add_local_tool(ping)
    inputs = []
    calls = 0

    async def stream_function(messages, info):
        nonlocal calls
        calls += 1
        if calls <= 2:
            client_input = normalize_client_input(
                ClientInput(payload=MessagePayload(content=f"steering {calls}"))
            )
            inputs.append(client_input)
            runtime._pending_user_inputs.append(client_input)
            runtime.notify_llm(f"notification {calls}")
            if use_tools:
                yield {0: DeltaToolCall(name="ping", json_args="{}")}
            else:
                yield f"reply {calls}"
        else:
            yield "done"

    async with runtime:
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            result = await runtime.run_turn("start")

        assert calls == 3
        assert context_resets(session) == []
        journal_messages = [
            entry.message
            for entry in session.journal
            if isinstance(entry, ModelMessageEvent)
        ]
        assert journal_messages == result.all_messages()
        assert session.message_history == result.all_messages()
        sequences = [
            entry.sequence
            for entry in session.journal
            if isinstance(entry, ModelMessageEvent)
        ]
        assert sequences == list(range(len(journal_messages)))

        forked = await session.fork_at(inputs[1].server_message_id)
        history = forked.message_history
        assert contains_input(history, inputs[0].server_message_id)
        assert not contains_input(history, inputs[1].server_message_id)
        assert any(
            isinstance(part, UserPromptPart) and part.content == "notification 1"
            for message in history
            for part in message.parts
        )
        # Persistence and the next turn must not duplicate messages already journaled.
        restored = AgentSession.model_validate_json(session.model_dump_json())
        assert restored.message_history == result.all_messages()
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            followup = await runtime.run_turn("follow-up")
        assert context_resets(session) == []
        assert [
            part for message in session.message_history for part in message.parts
        ] == [part for message in followup.all_messages() for part in message.parts]


@pytest.mark.asyncio
async def test_model_api_error_ends_turn_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "test"\n[agent.main]\nsystem_prompt = "Hi."\n')
    session = AgentSession(agent_name="main")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}), _StubIOAdapter(), session
    )
    requests = []

    async def stream_function(messages, info):
        requests.append(list(messages))
        raise ModelAPIError("test", "model unavailable")
        yield "unreachable"

    async with runtime:
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            with pytest.raises(ModelAPIError, match="model unavailable"):
                await runtime.run_turn("question")
    assert len(requests) == 1
    assert session.message_history == requests[0]
    assert context_resets(session) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("divergent", [False, True])
async def test_inference_reports_journal_mismatch_without_repair(
    tmp_path, monkeypatch, divergent
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "test"\n[agent.main]\nsystem_prompt = "Hi."\n')
    session = AgentSession(agent_name="main")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}), _StubIOAdapter(), session
    )
    request = ModelRequest.user_text_prompt("question")
    response = ModelResponse(parts=[TextPart(content="answer")])

    async def run(*args, **kwargs):
        session.record_model_message(
            ModelRequest.user_text_prompt("different") if divergent else request,
            run_id="run",
            sequence=0,
        )
        return SimpleNamespace(
            output="answer", new_messages=lambda: [request, response]
        )

    monkeypatch.setattr(runtime._pydantic_agent, "run", run)
    async with runtime:
        with pytest.raises(RuntimeError, match="journal diverged"):
            await runtime._run_inference("question", message_history=[])
    assert len(session.journal) == 1
    assert context_resets(session) == []
    assert len(session.message_history) == 1
    assert len(session.build_io_timeline()) == 1


@pytest.mark.asyncio
async def test_compaction_continuation_preserves_journal_and_fork_boundaries(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "test"\n[agent.main]\nsystem_prompt = "Hi."\n')
    session = AgentSession(agent_name="main")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}), _StubIOAdapter(), session
    )
    summary: list[ModelMessage] = [ModelRequest.user_text_prompt("private summary")]
    compacted = False

    async def compact(ctx, request_context):
        nonlocal compacted
        if not compacted:
            compacted = True
            compact_history(session, summary, False, "compact-context")
            raise ContinueAgentRun(summary)
        return request_context

    runtime.builtin_hooks.on.before_model_request(compact)

    async def stream_function(messages, info):
        yield "answer"

    async with runtime:
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            first = normalize_client_input(
                ClientInput(payload=MessagePayload(content="first"))
            )
            await runtime.run_turn(first)
            second = normalize_client_input(
                ClientInput(payload=MessagePayload(content="second"))
            )
            before_second = session.message_history
            await runtime.run_turn(second)
    assert len(context_resets(session)) == 1
    restored = AgentSession.model_validate_json(session.model_dump_json())
    assert restored.message_history == session.message_history
    assert (await restored.fork_at(first.server_message_id)).message_history == []
    assert (
        await restored.fork_at(second.server_message_id)
    ).message_history == before_second
    assert "private summary" not in str(restored.build_io_timeline())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_stream_failure_journals_partial_response(
    tmp_path, monkeypatch, cancelled
):
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    config_file = tmp_path / ".arox" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_ref = "test"\n[agent.main]\nsystem_prompt = "Hi."\n')
    session = AgentSession(agent_name="main")
    runtime = AgentRuntime(
        app_setup(cli_args={"workspace": str(tmp_path)}), _StubIOAdapter(), session
    )

    async def stream_function(messages, info):
        yield "partial answer"
        if cancelled:
            raise asyncio.CancelledError()
        raise ModelAPIError("test", "stream interrupted")

    async with runtime:
        with runtime._pydantic_agent.override(
            model=FunctionModel(stream_function=stream_function)
        ):
            with pytest.raises(asyncio.CancelledError if cancelled else ModelAPIError):
                await runtime.run_turn("question")
        await runtime.io_adapter.disconnect(runtime)
        replacement = IOEndpoint()
        runtime.agent_ep.pair(replacement)
        snapshot = await replacement.receive()
        assert isinstance(snapshot, SnapshotEvent)
        assert snapshot.journal_id == session.journal[-1].id
        assert any(
            isinstance(message, ModelResponse) and message.text == "partial answer"
            for message in session.build_io_timeline(through_id=snapshot.journal_id)
        )
        replacement.close()
    assert any(
        isinstance(message, ModelResponse) and message.text == "partial answer"
        for message in session.message_history
    )
    assert context_resets(session) == []
