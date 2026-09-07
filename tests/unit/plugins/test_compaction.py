import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from arox.core.agent_runtime import AgentRuntime, ContinueAgentRun
from arox.core.io import AbstractIOAdapter, AgentIOEndpoint, IOEndpoint, SnapshotEvent
from arox.core.session import AgentSession, ModelMessageEvent
from arox.core.turn import Turn
from arox.core.types import ClientInput, MessagePayload, normalize_client_input
from arox.plugins.compaction import CompactionEvent, CompactionPlugin
from arox.plugins.slots import PERSISTENT_CONTEXT
from tests.history import context_resets, record_messages, reset_history


class _TestIOAdapter(AbstractIOAdapter):
    async def handle_event(self, adapter_ep, event):
        pass


class _FakeAgent:
    def __init__(self, summary: str = "SUMMARY"):
        self.session = AgentSession(agent_name="compaction")
        self._summary = summary
        self.last_prompt = ""
        self.message_history = []
        self.name = "compaction"

    async def run(self, client_input: ClientInput):
        payload = client_input.payload
        assert isinstance(payload, MessagePayload)
        self.last_prompt = payload.text_content or ""
        return SimpleNamespace(output=self._summary)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def start_message(self, prompt):
        client_input = normalize_client_input(
            ClientInput(payload=MessagePayload(content=prompt))
        )
        return Turn(client_input, asyncio.create_task(self.run(client_input)))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setattr(
        "arox.plugins.compaction.AgentRuntime",
        lambda config_loader, _io_adapter, session: _prepare_fake_agent(
            config_loader._compaction_agent, session
        ),
    )


def _prepare_fake_agent(agent, session):
    agent.session = session
    return agent


class _MockAgent:
    """Minimal agent surface the CompactionPlugin touches."""

    def __init__(self, threshold: int | None, persistent=None):
        from arox.core.config import AgentConfig, Config

        self.model_config = None

        self.config = Config(
            compaction_threshold=threshold if threshold is not None else 0.7,
            agent={"compaction": AgentConfig(task_prompt="summary")},
        )
        if threshold is None:
            # Overwrite after instantiation if None is needed
            self.config.compaction_threshold = None  # type: ignore

        self.model_params = {}
        self.sent = []
        self.agent_ep: Any = SimpleNamespace(
            send=self._send,
        )
        self.history_lock = asyncio.Lock()
        self.session = AgentSession(agent_name="main")
        self.run_info = self.session.run_info
        self.run_info.llm_context_id = "ctx-original"
        self.workspace = "fake-workspace"

        self.io_adapter = _TestIOAdapter()
        self.config_loader = self

        self._stack = contextlib.AsyncExitStack()

        self._compaction_agent = _FakeAgent()
        self._persistent = persistent or []

    @property
    def message_history(self):
        return self.session.message_history

    @message_history.setter
    def message_history(self, value):
        reset_history(self.session, value)

    async def _send(self, msg):
        self.sent.append(msg)

    async def broadcast_session_tree(self):
        pass

    async def invoke_slot(self, slot, *args, **kwargs):
        if slot is PERSISTENT_CONTEXT:
            return [self._persistent] if self._persistent else []
        return None


def _plugin(agent: _MockAgent) -> CompactionPlugin:
    return CompactionPlugin(cast(AgentRuntime, agent))


def _ctx(total_tokens: int):
    return SimpleNamespace(
        usage=SimpleNamespace(total_tokens=total_tokens), run_id=None
    )


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _reply(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _first_text(message: ModelRequest) -> str:
    content = message.parts[0].content
    return content if isinstance(content, str) else str(content)


@pytest.mark.asyncio
async def test_auto_compaction_below_threshold_is_noop():
    agent = _MockAgent(threshold=100)
    agent.run_info.context_tokens = 50
    plugin = _plugin(agent)
    messages = [_user("a"), _reply("b"), _user("c")]

    out = await plugin.history_processor(_ctx(50), list(messages))

    assert out == messages
    assert [type(e) for e in agent.session.build_io_timeline()] == []
    assert agent.run_info.llm_context_id == "ctx-original"


@pytest.mark.asyncio
async def test_auto_compaction_disabled_without_threshold():
    agent = _MockAgent(threshold=None)
    agent.run_info.context_tokens = 10_000
    plugin = _plugin(agent)
    messages = [_user("a"), _reply("b"), _user("c")]

    out = await plugin.history_processor(_ctx(10_000), list(messages))

    assert out == messages
    assert agent.session.build_io_timeline() == ()


@pytest.mark.asyncio
async def test_auto_compaction_compacts_mid_tool_loop():
    """A trailing tool-return is folded into the summary, not orphaned.

    Compacting the whole history drops both the tool call and its return
    together, so no dangling tool_result is left behind.
    """
    agent = _MockAgent(threshold=100)
    agent.run_info.context_tokens = 500
    plugin = _plugin(agent)
    messages = [
        _user("question"),
        ModelResponse(
            parts=[ToolCallPart(tool_name="read", args={}, tool_call_id="1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read", content="x", tool_call_id="1")]
        ),
    ]

    with pytest.raises(ContinueAgentRun) as exc_info:
        await plugin.history_processor(_ctx(500), list(messages))
    out = exc_info.value.message_history

    # Everything collapsed into the summary; no leftover tool_return.
    assert agent._compaction_agent.session.agent_source == "compaction"
    compaction_history = agent._compaction_agent.session.message_history
    assert compaction_history == messages
    assert all(
        actual is not original
        for actual, original in zip(
            compaction_history,
            messages,
            strict=True,
        )
    )
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert not any(
        isinstance(p, ToolReturnPart)
        for m in out
        if isinstance(m, ModelRequest)
        for p in m.parts
    )
    assert [type(e) for e in agent.session.build_io_timeline()] == [CompactionEvent]
    compaction = cast(CompactionEvent, agent.session.build_io_timeline()[0])
    assert compaction.trigger == "token_threshold"
    assert agent.sent == [compaction]
    assert agent.run_info.llm_context_id != "ctx-original"
    assert agent.run_info.context_tokens == 0


@pytest.mark.asyncio
async def test_auto_compaction_preserves_events_for_reconnect_replay():
    agent = _MockAgent(threshold=100)
    agent.run_info.context_tokens = 500
    agent.agent_ep = AgentIOEndpoint()
    await agent.agent_ep.send("tool output before compaction")
    plugin = _plugin(agent)

    with pytest.raises(ContinueAgentRun):
        await plugin.history_processor(_ctx(500), [_user("question")])

    replacement = IOEndpoint()
    agent.agent_ep.pair(replacement)
    assert await replacement.receive() == SnapshotEvent(None)
    start = await replacement.receive()
    end = await replacement.receive()
    assert start.part.content == "tool output before compaction"
    assert end.part.content == "tool output before compaction"


@pytest.mark.asyncio
async def test_auto_compaction_records_event_and_stays_consistent():
    # Persistent context makes the recorded base (compacted[:-1]) non-trivial:
    # file.py emits a [ModelResponse(read call), ModelRequest(read return)] pair.
    persistent = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="read", args={}, tool_call_id="p")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read", content="ctx", tool_call_id="p")]
        ),
    ]
    agent = _MockAgent(threshold=100, persistent=persistent)
    agent.run_info.context_tokens = 500
    plugin = _plugin(agent)
    messages = [_user("old 1"), _reply("old reply 1"), _user("current question")]
    agent.message_history = messages[:-1]

    with pytest.raises(ContinueAgentRun) as exc_info:
        await plugin.history_processor(_ctx(500), list(messages))
    out = exc_info.value.message_history

    # Whole history compacted to: summary + persistent context.
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert len(out) == 3

    # Context replacement and its visible marker are committed together.
    events = agent.session.build_io_timeline()
    assert [type(e) for e in events] == [CompactionEvent]
    compaction = cast(CompactionEvent, events[0])
    assert compaction.trigger == "token_threshold"
    assert agent.run_info.llm_context_id != "ctx-original"
    assert compaction.llm_context_id == agent.run_info.llm_context_id

    compaction_index = agent.session.index_of_event(compaction.id)
    assert compaction_index is not None
    replacement_events = agent.session.journal[compaction_index:]
    assert [event.event_type for event in replacement_events] == [
        "compaction",
        "context_reset",
        "model_message",
        "model_message",
        "model_message",
    ]
    assert "compaction" not in replacement_events[1].model_dump()
    assert "messages" not in replacement_events[1].model_dump()
    assert [
        event.message
        for event in replacement_events
        if isinstance(event, ModelMessageEvent)
    ] == out
    assert all(
        event.context_only
        for event in replacement_events
        if isinstance(event, ModelMessageEvent)
    )
    assert not hasattr(compaction, "compacted_messages")
    assert agent.session.message_history == out

    # Subsequent messages append after the reset.
    response = _reply("answer")
    complete_history = [*out, response]

    record_messages(agent.session, [response])
    assert agent.session.message_history == complete_history


@pytest.mark.asyncio
async def test_compact_tool_defers_compaction_until_history_processing():
    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    messages = [_user("old"), _reply("reply")]

    result = plugin.compact(" Preserve implementation details. ")

    assert result == "Conversation history compaction requested."
    assert agent.session.build_io_timeline() == ()
    assert agent.message_history == []

    with pytest.raises(ContinueAgentRun) as exc_info:
        await plugin.history_processor(_ctx(0), messages)
    out = exc_info.value.message_history

    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert agent._compaction_agent.last_prompt == (
        "summary\n\nAdditional instructions: Preserve implementation details."
    )
    assert [type(e) for e in agent.session.build_io_timeline()] == [CompactionEvent]
    compaction = cast(CompactionEvent, agent.session.build_io_timeline()[0])
    assert compaction.trigger == "tool_request"
    assert agent.sent == [compaction]

    second_out = await plugin.history_processor(_ctx(0), out)

    assert second_out is out
    assert [type(e) for e in agent.session.build_io_timeline()] == [CompactionEvent]


@pytest.mark.asyncio
async def test_manual_compact_records_event_and_replaces_history():
    from arox.plugins.compaction import CompactEvent

    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    agent.message_history = [_user("old"), _reply("reply")]

    result = await plugin.handle_compact(CompactEvent(extra_instructions=""))

    assert result == "Conversation history compacted successfully."
    assert [type(e) for e in agent.session.build_io_timeline()] == [CompactionEvent]
    compaction = cast(CompactionEvent, agent.session.build_io_timeline()[0])
    assert compaction.trigger == "manual"
    assert agent.sent == [compaction]
    assert agent.run_info.llm_context_id != "ctx-original"
    # Manual compaction replaces the live history with the summary base.
    assert "SUMMARY" in _first_text(agent.message_history[0])
    assert len(agent.message_history) == 1


@pytest.mark.asyncio
async def test_manual_compact_waits_for_history_lock_and_uses_latest_history():
    from arox.plugins.compaction import CompactEvent

    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    agent.message_history = [_user("old")]
    await agent.history_lock.acquire()

    task = asyncio.create_task(
        plugin.handle_compact(CompactEvent(extra_instructions=""))
    )
    await asyncio.sleep(0)
    assert agent.session.build_io_timeline() == ()

    latest_history = [_user("old"), _reply("new answer")]
    agent.message_history = latest_history
    agent.history_lock.release()

    assert await task == "Conversation history compacted successfully."
    previous_reset = context_resets(agent.session)[-2]
    previous_index = agent.session.index_of_event(previous_reset.id)
    assert previous_index is not None
    assert [
        event.message
        for event in agent.session.journal[previous_index + 1 : previous_index + 3]
        if isinstance(event, ModelMessageEvent)
    ] == latest_history


def test_token_threshold_is_recomputed_after_config_change():
    agent = _MockAgent(threshold=100)
    plugin = _plugin(agent)

    assert plugin._resolve_token_threshold() == 100

    agent.config.compaction_threshold = 250

    assert plugin._resolve_token_threshold() == 250
