import asyncio
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.messages import (
    BinaryContent,
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

from arox.core.agent_runtime import AgentDeps, AgentRuntime
from arox.core.app import app_setup
from arox.core.background import BackgroundTaskBroker
from arox.core.io import AbstractIOAdapter, RequestEvent
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
    CommandRequestedEvent,
)
from arox.core.types import TurnStateEvent, UserInput, UserMessageEvent
from arox.plugins.core import SetModelEvent


class _StubIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_ep, event):
        pass


class _FailingToolPlugin(Plugin):
    @tool()
    def fail(self) -> None:
        raise RuntimeError("expected failure")


@pytest.mark.asyncio
async def test_command_dispatch_records_request_and_completion_timeline():
    session = AgentSession(agent_name="main")

    async def dispatch(command):
        assert command == "/info"
        return CommandDispatchResult(
            "handled",
            CommandReply(req_id="reply", output="details"),
        )

    class Endpoint:
        def __init__(self):
            self.sent = []
            self.snapshot_value = None

        async def send(self, event):
            self.sent.append(event)

        def snapshot(self, value):
            self.snapshot_value = value

    endpoint = Endpoint()
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(
            session=session,
            command_manager=SimpleNamespace(dispatch=dispatch),
            agent_ep=endpoint,
        ),
    )

    result = await AgentRuntime._dispatch_command(runtime, "/info")

    assert result.status == "handled"
    assert endpoint.sent == ["details"]
    assert len(session.events) == 2
    requested, completed = session.events
    assert isinstance(requested, CommandRequestedEvent)
    assert requested.display_text == "/info"
    assert isinstance(completed, CommandCompletedEvent)
    assert completed.command_event_id == requested.id
    assert completed.output == "details"
    assert endpoint.snapshot_value == session.build_io_snapshot()


@pytest.mark.asyncio
async def test_llm_notifications_are_injected_once_before_model_request():
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.background_tasks = BackgroundTaskBroker()
    runtime._pending_user_inputs = deque()
    runtime.message_history_fallback = []
    runtime.notify_llm("First task finished.")
    runtime.notify_llm("Second task finished.")
    request_context = SimpleNamespace(messages=[])

    await runtime._before_model_request(None, request_context)

    assert len(request_context.messages) == 1
    notice = request_context.messages[0]
    assert isinstance(notice, ModelRequest)
    assert notice.parts[0].content == ("First task finished.\n\nSecond task finished.")
    assert not runtime.background_tasks.drain_notices()

    await runtime._before_model_request(None, request_context)
    assert len(request_context.messages) == 1


@pytest.mark.asyncio
async def test_run_error_logs_exception_traceback(caplog):
    runtime = cast(
        AgentRuntime,
        SimpleNamespace(message_history_fallback=[], new_message_index=0),
    )
    ctx = cast(
        RunContext[AgentDeps],
        SimpleNamespace(
            usage=RunUsage(),
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
    session.replace_message_history(
        [
            ModelRequest(parts=[internal_user_prompt_part("<file>secret</file>")]),
            ModelRequest(parts=[UserPromptPart(content="first question")]),
            ModelResponse(parts=[TextPart(content="first answer")]),
        ]
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

    first_request = session.message_history.messages[0]
    assert isinstance(first_request, ModelRequest)
    assert not first_request.metadata
    internal_content = first_request.parts[0].content
    assert isinstance(internal_content, list)
    assert isinstance(internal_content[0], TextContent)
    assert internal_content[0].metadata == {AROX_INTERNAL_KEY: True}

    visible = visible_message_history(session.message_history.messages)
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
async def test_request_event_dispatches_to_handler(tmp_path, monkeypatch):
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

    class CustomEvent(RequestEvent):
        pass

    received: list[RequestEvent] = []

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
        event for event in sent_events if isinstance(event, UserMessageEvent)
    )
    assert user_message.user_input.text_content == "fail"
    session_snapshot = runtime.session.build_io_snapshot()
    assert len(session_snapshot) == 2
    assert isinstance(session_snapshot[0], ModelRequest)
    assert isinstance(session_snapshot[1], ModelResponse)
    assert session_snapshot[1].text == "RuntimeError: model failed"

    endpoint_snapshot = runtime.agent_ep._snapshot_value
    assert len(endpoint_snapshot) == 1
    assert isinstance(endpoint_snapshot[0], ModelRequest)


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
    assert all(event.event_type != "error" for event in runtime.session.events)


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

    async def blocking_turn(user_input):
        consumed_inputs.append(user_input.text_content)
        started.set()
        await release.wait()
        return SimpleNamespace(output=user_input)

    runtime._run_turn_input = blocking_turn  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    async with runtime:
        turn = await runtime.accept_input("work")
        assert turn is not None
        task = turn.task
        await started.wait()
        assert runtime.turn is turn
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.01)
        parallel_turn = await runtime.accept_input("parallel work")
        assert parallel_turn is turn

        assert await runtime.cancel_turn()
        assert task.cancelled()
        assert not await runtime.cancel_turn()
        assert turn.result is None
        assert isinstance(turn.error, asyncio.CancelledError)
        assert consumed_inputs == ["work"]

        release.set()
        completed_turn = await runtime.accept_input("completed work")
        assert completed_turn is not None
        completed_result = await completed_turn
        assert isinstance(completed_result.output, UserInput)
        assert completed_result.output.text_content == "completed work"
        assert runtime.turn is completed_turn
        assert completed_turn.result is completed_result
        assert completed_turn.error is None

        started.clear()
        release.clear()
        consumed_inputs.clear()
        queued_turn = await runtime.accept_input("first queued work")
        assert queued_turn is not None
        await started.wait()
        assert await runtime.accept_input("second queued work") is queued_turn
        request_context = SimpleNamespace(messages=[])
        await runtime._before_model_request(
            cast(RunContext[AgentDeps], None),
            cast(ModelRequestContext, request_context),
        )
        injected = request_context.messages[0]
        assert isinstance(injected, ModelRequest)
        injected_content = injected.parts[0].content
        assert not isinstance(injected_content, str)
        assert isinstance(injected_content[0], TextContent)
        assert injected_content[0].content == "second queued work"
        assert not runtime._pending_user_inputs
        release.set()
        queued_result = await queued_turn
        assert consumed_inputs == ["first queued work"]
        assert queued_result.output.text_content == "first queued work"

        started.clear()
        release.clear()
        consumed_inputs.clear()
        trailing_turn = await runtime.accept_input("last model request")
        assert trailing_turn is not None
        await started.wait()
        assert await runtime.accept_input("missed injection window") is trailing_turn
        release.set()
        trailing_result = await trailing_turn
        assert consumed_inputs == ["last model request", "missed injection window"]
        assert trailing_result.output.text_content == "missed injection window"

        async def failed_turn(user_input):
            raise RuntimeError("model failed")

        runtime._run_turn_input = failed_turn  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
        with pytest.raises(RuntimeError, match="model failed"):
            failed_turn_handle = await runtime.accept_input("failed work")
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
    assert session.events[-1].event_type == "error"
    assert "RuntimeError: something broke" in session.events[-1].error
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
    assert session.events[-1].event_type == "error"
    assert session.events[-1].error == "Task interrupted."
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
