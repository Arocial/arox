import contextlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from arox.core.agent_runtime import AgentRuntime
from arox.core.session import AgentSession, CompactionEvent
from arox.plugins.compaction import CompactionPlugin
from arox.plugins.slots import PERSISTENT_CONTEXT


class _FakeAgent:
    def __init__(self, summary: str = "SUMMARY"):
        self.session = AgentSession(agent_name="compaction")
        self._summary = summary
        self.last_prompt = ""
        self.message_history = []
        self.name = "compaction"

    async def run(self, user_input=None):
        self.last_prompt = str(user_input or "")
        return SimpleNamespace(output=self._summary)


class _FakeTaskRunner:
    def __init__(self, session, config_loader, io_adapter):
        self.session = session
        self.runtime = config_loader._compaction_agent

    async def __aenter__(self):
        self.runtime.session = self.session
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def run(self, prompt, *, render_user_message=False):
        return self.runtime.run(prompt)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setattr("arox.core.agent_runtime.AgentRuntime.__aenter__", AsyncMock())
    monkeypatch.setattr("arox.core.agent_runtime.AgentRuntime.__aexit__", AsyncMock())
    monkeypatch.setattr("arox.plugins.compaction.TaskRunner", _FakeTaskRunner)


class _MockAgent:
    """Minimal agent surface the CompactionPlugin touches."""

    def __init__(self, threshold: int | None, persistent=None):
        from arox.core.config import AgentConfig, Config

        self.run_info = SimpleNamespace(context_tokens=0, llm_context_id="ctx-original")
        self.model_config = None

        self.config = Config(
            compaction_threshold=threshold if threshold is not None else 0.7,
            agent={"compaction": AgentConfig(task_prompt="summary")},
        )
        if threshold is None:
            # Overwrite after instantiation if None is needed
            self.config.compaction_threshold = None  # type: ignore

        self.model_params = {}
        self.agent_ep = SimpleNamespace(send=self._send)
        self.session = AgentSession(agent_name="main")
        self.workspace = "fake-workspace"

        self.io_adapter = SimpleNamespace(
            handle_event=AsyncMock(),
            on_endpoint_closed=AsyncMock(),
        )
        self.config_loader = self

        self._stack = contextlib.AsyncExitStack()

        self._compaction_agent = _FakeAgent()
        self._persistent = persistent or []

    @property
    def message_history(self):
        return self.session.message_history.messages

    @message_history.setter
    def message_history(self, value):
        self.session.replace_message_history(value)

    async def _send(self, _msg):
        return None

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
    assert [e.event_type for e in agent.session.events] == []
    assert agent.run_info.llm_context_id == "ctx-original"


@pytest.mark.asyncio
async def test_auto_compaction_disabled_without_threshold():
    agent = _MockAgent(threshold=None)
    agent.run_info.context_tokens = 10_000
    plugin = _plugin(agent)
    messages = [_user("a"), _reply("b"), _user("c")]

    out = await plugin.history_processor(_ctx(10_000), list(messages))

    assert out == messages
    assert agent.session.events == []


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

    out = await plugin.history_processor(_ctx(500), list(messages))

    # Everything collapsed into the summary; no leftover tool_return.
    assert agent._compaction_agent.session.agent_source == "compaction"
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert not any(
        isinstance(p, ToolReturnPart)
        for m in out
        if isinstance(m, ModelRequest)
        for p in m.parts
    )
    assert [e.event_type for e in agent.session.events] == ["compaction"]
    assert agent.run_info.llm_context_id != "ctx-original"


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

    out = await plugin.history_processor(_ctx(500), list(messages))

    # Whole history compacted to: summary + persistent context.
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert len(out) == 3

    # A compaction event is recorded without duplicating the compacted messages;
    # the compacted output becomes the active history segment.
    events = agent.session.events
    assert [e.event_type for e in events] == ["compaction"]
    compaction = cast(CompactionEvent, events[0])
    assert agent.run_info.llm_context_id != "ctx-original"
    assert compaction.llm_context_id == agent.run_info.llm_context_id

    assert not hasattr(compaction, "compacted_messages")
    assert agent.session.message_history.messages == out
    assert len(agent.session.archived_message_histories) == 1
    assert agent.session.archived_message_histories[0].messages == messages

    # The completed step updates only the active segment.
    response = _reply("answer")
    complete_history = [*out, response]

    agent.session.record_step(complete_history)
    assert agent.session.message_history.messages == complete_history


@pytest.mark.asyncio
async def test_compact_tool_defers_compaction_until_history_processing():
    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    messages = [_user("old"), _reply("reply")]

    result = plugin.compact(" Preserve implementation details. ")

    assert result == "Conversation history compaction requested."
    assert agent.session.events == []
    assert agent.message_history == []

    out = await plugin.history_processor(_ctx(0), messages)

    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert agent._compaction_agent.last_prompt == (
        "summary\n\nAdditional instructions: Preserve implementation details."
    )
    assert [e.event_type for e in agent.session.events] == ["compaction"]

    second_out = await plugin.history_processor(_ctx(0), out)

    assert second_out is out
    assert [e.event_type for e in agent.session.events] == ["compaction"]


@pytest.mark.asyncio
async def test_manual_compact_records_event_and_replaces_history():
    from arox.plugins.compaction import CompactEvent

    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    agent.message_history = [_user("old"), _reply("reply")]

    await plugin.handle_compact(CompactEvent(extra_instructions=""))

    assert [e.event_type for e in agent.session.events] == ["compaction"]
    assert agent.run_info.llm_context_id != "ctx-original"
    # Manual compaction replaces the live history with the summary base.
    assert "SUMMARY" in _first_text(agent.message_history[0])
    assert len(agent.message_history) == 1
