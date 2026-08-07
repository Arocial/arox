import asyncio
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

from arox.core.llm_base import LLMBaseAgent
from arox.core.session import AgentSession, CompactionEvent, StepEvent
from arox.plugins.compaction import CompactionAgent, CompactionPlugin
from arox.plugins.slots import PERSISTENT_CONTEXT, SUBAGENTS


class _FakeCompactionAgent(CompactionAgent):
    """Passes the ``isinstance(sub, CompactionAgent)`` check without a full init."""

    def __init__(self, summary: str = "SUMMARY"):
        self.session = AgentSession(agent_name="compaction")
        self._summary = summary
        self.last_extra_instructions = ""
        self.message_history = []
        self.name = "compaction"

    async def execute_task(self, task: str) -> str | None:
        return await self.summarize(self.message_history, extra_instructions=task)

    async def summarize(self, messages, extra_instructions: str = "") -> str:
        self.last_extra_instructions = extra_instructions
        return self._summary


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setattr(
        "arox.plugins.compaction.CompactionAgent.summarize",
        AsyncMock(return_value="SUMMARY"),
    )
    monkeypatch.setattr("arox.core.llm_base.LLMBaseAgent.__aenter__", AsyncMock())
    monkeypatch.setattr("arox.core.llm_base.LLMBaseAgent.__aexit__", AsyncMock())


class _MockAgent:
    """Minimal agent surface the CompactionPlugin touches."""

    def __init__(self, threshold: int | None, persistent=None):
        from arox.core.config import AgentConfig, Config

        self.message_history = []
        self.run_info = SimpleNamespace(context_tokens=0, llm_context_id="ctx-original")
        self.model_config = None

        self.config = Config(
            compaction_threshold=threshold if threshold is not None else 0.7,
            agent={"compaction": AgentConfig(type="compaction", task_prompt="summary")},
        )
        if threshold is None:
            # Overwrite after instantiation if None is needed
            self.config.compaction_threshold = None  # type: ignore

        self.model_params = {}
        self.agent_io = SimpleNamespace(send=self._send)
        self.session = AgentSession(agent_name="main")
        self.workspace = "fake-workspace"

        async def _fake_process_io(adapter_io):
            pass

        self.io_adapter = SimpleNamespace(
            register_host=AsyncMock(), _process_io=_fake_process_io
        )

        self._stack = contextlib.AsyncExitStack()

        self._compaction_agent = _FakeCompactionAgent()
        self._persistent = persistent or []

    async def _send(self, _msg):
        return None

    async def broadcast_agent_info(self):
        pass

    async def invoke_slot(self, slot, *args, **kwargs):
        if slot is SUBAGENTS:
            return [self._compaction_agent]
        if slot is PERSISTENT_CONTEXT:
            return [self._persistent] if self._persistent else []
        from arox.plugins.slots import RUN_SUBAGENT

        if slot is RUN_SUBAGENT:
            callback = kwargs.get("on_subagent_created")
            if callback:
                result = callback(self._compaction_agent)
                if asyncio.iscoroutine(result):
                    await result
            return await self._compaction_agent.run_task(
                args[1] if len(args) > 1 else ""
            )
        return None


def _plugin(agent: _MockAgent) -> CompactionPlugin:
    return CompactionPlugin(cast(LLMBaseAgent, agent))


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
    assert agent.run_info.llm_context_id != "ctx-original"
    assert compaction.llm_context_id == agent.run_info.llm_context_id

    assert compaction.compacted_messages == out

    # Since step_boundary=False clears the base history during rebuild, the
    # subsequent agent_step provides the full post-compaction history.
    response = _reply("answer")

    agent.session.add_event(StepEvent(new_messages=[*out, response]))
    rebuilt = agent.session.rebuild_message_history()
    assert rebuilt == [*out, response]


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
    assert agent._compaction_agent.last_extra_instructions == (
        "Preserve implementation details."
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
