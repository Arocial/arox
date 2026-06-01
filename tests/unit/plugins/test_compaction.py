from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from arox.core.llm_base import LLMBaseAgent
from arox.core.session import AgentSession, CompactionEvent
from arox.plugins.compaction import CompactionAgent, CompactionPlugin
from arox.plugins.slots import PERSISTENT_CONTEXT, SUBAGENTS


class _FakeCompactionAgent(CompactionAgent):
    """Passes the ``isinstance(sub, CompactionAgent)`` check without a full init."""

    def __init__(self, summary: str = "SUMMARY"):
        self.session = type("DummySession", (), {"agent_name": "compaction"})()
        self._summary = summary

    async def summarize(self, messages, extra_instructions: str = "") -> str:
        return self._summary


class _MockAgent:
    """Minimal agent surface the CompactionPlugin touches."""

    def __init__(self, threshold: int | None, persistent=None):
        self.message_history = []
        self.llm_context_id = "ctx-original"
        self.model_config = None
        self.parsed_config = SimpleNamespace(compaction_threshold=threshold)
        self.model_params = {}
        self.agent_io = SimpleNamespace(send=self._send)
        self.session = AgentSession(agent_name="main")
        self._compaction_agent = _FakeCompactionAgent()
        self._persistent = persistent or []

    async def _send(self, _msg):
        return None

    async def invoke_slot(self, slot, *args, **kwargs):
        if slot is SUBAGENTS:
            return [self._compaction_agent]
        if slot is PERSISTENT_CONTEXT:
            return [self._persistent] if self._persistent else []
        return None


def _plugin(agent: _MockAgent) -> CompactionPlugin:
    return CompactionPlugin(cast(LLMBaseAgent, agent))


def _ctx(total_tokens: int):
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=total_tokens))


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
    plugin = _plugin(agent)
    messages = [_user("a"), _reply("b"), _user("c")]

    out = await plugin.history_processor(_ctx(50), list(messages))

    assert out == messages
    assert [e.event_type for e in agent.session.events] == []
    assert agent.llm_context_id == "ctx-original"


@pytest.mark.asyncio
async def test_auto_compaction_disabled_without_threshold():
    agent = _MockAgent(threshold=None)
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
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert not any(
        isinstance(p, ToolReturnPart)
        for m in out
        if isinstance(m, ModelRequest)
        for p in m.parts
    )
    assert [e.event_type for e in agent.session.events] == ["compaction"]
    assert agent.llm_context_id != "ctx-original"


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
    plugin = _plugin(agent)
    messages = [_user("old 1"), _reply("old reply 1"), _user("current question")]

    out = await plugin.history_processor(_ctx(500), list(messages))

    # Whole history compacted to: summary + persistent context.
    assert isinstance(out[0], ModelRequest)
    assert "SUMMARY" in _first_text(out[0])
    assert len(out) == 3

    # A compaction event was recorded, cache key rotated, and the recorded base
    # excludes the last compacted message (it is captured by this step's
    # agent_step instead).
    events = agent.session.events
    assert [e.event_type for e in events] == ["compaction"]
    compaction = cast(CompactionEvent, events[0])
    assert agent.llm_context_id != "ctx-original"
    assert compaction.llm_context_id == agent.llm_context_id

    assert compaction.compacted_messages == out

    # Since step_boundary=False clears the base history during rebuild, the
    # subsequent agent_step provides the full post-compaction history.
    response = _reply("answer")

    agent.session.add_event(
        "agent_step",
        {"new_messages": [*out, response]},
    )
    rebuilt = agent.session.rebuild_message_history()
    assert rebuilt == [*out, response]


@pytest.mark.asyncio
async def test_manual_compact_records_event_and_replaces_history():
    from arox.plugins.compaction import CompactEvent

    agent = _MockAgent(threshold=None)
    plugin = _plugin(agent)
    agent.message_history = [_user("old"), _reply("reply")]

    await plugin.handle_compact(CompactEvent(extra_instructions=""))

    assert [e.event_type for e in agent.session.events] == ["compaction"]
    assert agent.llm_context_id != "ctx-original"
    # Manual compaction replaces the live history with the summary base.
    assert "SUMMARY" in _first_text(agent.message_history[0])
    assert len(agent.message_history) == 1
