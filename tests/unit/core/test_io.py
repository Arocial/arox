import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pydantic_ai import (
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

from arox.core.io import AbstractIOAdapter, AgentIOEndpoint, IOEndpoint, SnapshotEvent
from arox.core.types import TurnStateEvent

if TYPE_CHECKING:
    from arox.core.agent_runtime import AgentRuntime


class _RecordingAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.events: asyncio.Queue = asyncio.Queue()
        self.closed_endpoints = []

    async def handle_event(self, adapter_ep, event):
        await self.events.put((adapter_ep, event))

    async def on_endpoint_closed(self, adapter_ep):
        self.closed_endpoints.append(adapter_ep)


async def _drain(endpoint, on_event):
    """Continuously receive on ``endpoint`` and forward events."""
    while True:
        try:
            event = await endpoint.receive()
        except Exception:
            return
        await on_event(event)


def _create_test_channel():
    agent_ep = IOEndpoint()
    adapter_ep = IOEndpoint()
    agent_ep.pair(adapter_ep)
    return agent_ep, adapter_ep


@pytest.mark.asyncio
async def test_fire_and_forget_send_returns_none():
    @dataclass
    class Plain:
        value: int = 0

    agent_ep, adapter_ep = _create_test_channel()
    async with agent_ep, adapter_ep:
        result = await adapter_ep.send(Plain(value=42))
        assert result is None
        received = await asyncio.wait_for(agent_ep.receive(), timeout=1.0)
        assert isinstance(received, Plain)
        assert received.value == 42


@pytest.mark.asyncio
async def test_agent_endpoint_dispatches_plain_events():
    @dataclass
    class Plain:
        value: int

    agent_ep = AgentIOEndpoint()
    adapter_ep = IOEndpoint()
    agent_ep.pair(adapter_ep)
    received: asyncio.Queue[int] = asyncio.Queue()
    result = agent_ep.register_event_handler(
        Plain, lambda event: received.put(event.value)
    )

    async with agent_ep, adapter_ep:
        await adapter_ep.send(Plain(42))
        assert await asyncio.wait_for(received.get(), timeout=1) == 42

    assert result is None


@pytest.mark.asyncio
async def test_close_stops_receiving():
    endpoint = IOEndpoint()
    endpoint.close()
    with pytest.raises(StopAsyncIteration):
        await endpoint.receive()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(endpoint.receive(), timeout=1)


def test_close_disconnects_without_closing_peer():
    endpoint = IOEndpoint()
    peer = IOEndpoint()
    endpoint.pair(peer)

    endpoint.close()

    assert endpoint.peer is None
    assert peer.peer is None
    assert not peer._closed

    replacement = IOEndpoint()
    peer.pair(replacement)
    assert peer.peer is replacement


@pytest.mark.asyncio
async def test_adapter_consumes_peer_events_and_cleans_up():
    adapter = _RecordingAdapter()
    agent_ep = AgentIOEndpoint()
    runtime = cast("AgentRuntime", SimpleNamespace(agent_ep=agent_ep))
    assert not adapter.adapter_ep_to_runtime

    async with agent_ep:
        assert not adapter.adapter_ep_to_runtime
        adapter_ep = await adapter.connect(runtime)
        _, snapshot = await asyncio.wait_for(adapter.events.get(), timeout=1)
        assert snapshot == SnapshotEvent(None)
        assert adapter.agent_io_for(adapter_ep) is runtime
        await agent_ep.send("hello")
        endpoint, start_event = await asyncio.wait_for(adapter.events.get(), timeout=1)
        _, end_event = await asyncio.wait_for(adapter.events.get(), timeout=1)

        assert endpoint is adapter_ep
        assert start_event.part.content == "hello"
        assert end_event.part.content == "hello"
        await adapter.disconnect(runtime)

    assert adapter.closed_endpoints == [adapter_ep]
    assert adapter_ep not in adapter.adapter_ep_to_runtime


@pytest.mark.asyncio
async def test_pair_replays_snapshot_and_cached_events_then_streams_live_events():
    endpoint = AgentIOEndpoint()
    await endpoint.send("cached")

    peer = IOEndpoint()
    endpoint.pair(peer)
    assert endpoint.peer is peer
    assert peer.peer is endpoint
    assert await peer.receive() == SnapshotEvent(None)
    assert (await peer.receive()).part.content == "cached"
    assert (await peer.receive()).part.content == "cached"

    await endpoint.send("live")
    assert (await peer.receive()).part.content == "live"

    replacement = IOEndpoint()
    with pytest.raises(RuntimeError, match="already connected"):
        endpoint.pair(replacement)
    endpoint.disconnect()
    assert peer.peer is None
    assert endpoint.peer is None
    endpoint.pair(replacement)
    assert endpoint.peer is replacement
    assert replacement.peer is endpoint
    assert await replacement.receive() == SnapshotEvent(None)
    assert [(await replacement.receive()).part.content for _ in range(4)] == [
        "cached",
        "cached",
        "live",
        "live",
    ]


@pytest.mark.asyncio
async def test_snapshot_is_only_sent_when_peer_connects():
    endpoint = AgentIOEndpoint()
    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(None)

    await endpoint.send(TurnStateEvent(busy=False))
    assert await peer.receive() == TurnStateEvent(busy=False)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(peer.receive(), timeout=0.01)

    endpoint.checkpoint("completed")
    endpoint.disconnect()
    replacement = IOEndpoint()
    endpoint.pair(replacement)

    assert await replacement.receive() == SnapshotEvent("completed")


@pytest.mark.asyncio
async def test_adapter_connect_replaces_old_peer():
    adapter = _RecordingAdapter()
    agent_ep = AgentIOEndpoint()
    runtime = cast("AgentRuntime", SimpleNamespace(agent_ep=agent_ep))
    old_peer = await adapter.connect(runtime)
    new_peer = await adapter.connect(runtime)

    assert old_peer._closed
    assert old_peer not in adapter.adapter_ep_to_runtime
    assert old_peer not in adapter._event_consumer_tasks

    await agent_ep.send("new")
    received = []
    while len(received) < 2:
        endpoint, event = await asyncio.wait_for(adapter.events.get(), timeout=1)
        if endpoint is new_peer and not isinstance(event, SnapshotEvent):
            received.append(event)
    assert [event.part.content for event in received] == ["new", "new"]
    await adapter.disconnect(runtime)


@pytest.mark.asyncio
async def test_checkpoint_replaces_replay_without_changing_connected_inbox():
    endpoint = AgentIOEndpoint()
    connected = IOEndpoint()
    endpoint.pair(connected)
    assert await connected.receive() == SnapshotEvent(None)
    part = TextPart(content="completed")
    await endpoint.send(TurnStateEvent(busy=True))
    await endpoint.send(PartStartEvent(index=0, part=part))
    await endpoint.send(PartEndEvent(index=0, part=part))
    await endpoint.send(TurnStateEvent(busy=False))
    endpoint.checkpoint("completed")

    assert await connected.receive() == TurnStateEvent(busy=True)
    assert isinstance(await connected.receive(), PartStartEvent)
    assert isinstance(await connected.receive(), PartEndEvent)
    assert await connected.receive() == TurnStateEvent(busy=False)
    endpoint.disconnect()

    replacement = IOEndpoint()
    endpoint.pair(replacement)
    assert await replacement.receive() == SnapshotEvent("completed")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(replacement.receive(), timeout=0.01)


@pytest.mark.asyncio
async def test_checkpoint_replays_only_the_live_tail():
    endpoint = AgentIOEndpoint()
    await endpoint.send("discarded")
    endpoint.checkpoint("completed")
    pending = PartStartEvent(index=0, part=TextPart(content="new partial"))
    await endpoint.send(pending)

    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent("completed")
    assert await peer.receive() is pending


@pytest.mark.asyncio
async def test_replay_compacts_part_deltas_without_changing_live_stream():
    endpoint = AgentIOEndpoint()
    connected = IOEndpoint()
    endpoint.pair(connected)
    assert await connected.receive() == SnapshotEvent(None)

    start = PartStartEvent(index=0, part=TextPart(content="Hel"))
    first_delta = PartDeltaEvent(
        index=0,
        delta=TextPartDelta(content_delta="lo"),
    )
    second_delta = PartDeltaEvent(
        index=0,
        delta=TextPartDelta(content_delta=" world"),
    )
    end = PartEndEvent(index=0, part=TextPart(content="Hello world"))

    for event in (start, first_delta, second_delta, end):
        await endpoint.send(event)

    assert [await connected.receive() for _ in range(4)] == [
        start,
        first_delta,
        second_delta,
        end,
    ]

    endpoint.disconnect()
    replacement = IOEndpoint()
    endpoint.pair(replacement)

    assert await replacement.receive() == SnapshotEvent(None)
    replay_start = await replacement.receive()
    replay_end = await replacement.receive()
    assert isinstance(replay_start, PartStartEvent)
    assert replay_start.part == TextPart(content="Hello world")
    assert replay_end is end
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(replacement.receive(), timeout=0.01)


@pytest.mark.asyncio
async def test_replay_compacts_incomplete_thinking_and_tool_call_parts():
    endpoint = AgentIOEndpoint()
    await endpoint.send(PartStartEvent(index=0, part=ThinkingPart(content="Consider")))
    await endpoint.send(
        PartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(content_delta=" this"),
        )
    )
    await endpoint.send(
        PartStartEvent(
            index=1,
            part=ToolCallPart("search", '{"query":', "call-1"),
        )
    )
    await endpoint.send(
        PartDeltaEvent(
            index=1,
            delta=ToolCallPartDelta(args_delta='"pydantic"}'),
        )
    )

    peer = IOEndpoint()
    endpoint.pair(peer)

    assert await peer.receive() == SnapshotEvent(None)
    thinking = await peer.receive()
    tool_call = await peer.receive()
    assert isinstance(thinking, PartStartEvent)
    assert thinking.part == ThinkingPart(content="Consider this")
    assert isinstance(tool_call, PartStartEvent)
    assert tool_call.part == ToolCallPart(
        "search",
        '{"query":"pydantic"}',
        "call-1",
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(peer.receive(), timeout=0.01)


@pytest.mark.asyncio
async def test_non_stream_event_starts_a_new_replay_compaction_segment():
    endpoint = AgentIOEndpoint()
    completed = TextPart(content="completed")
    pending = TextPart(content="pending")
    boundary = TurnStateEvent(busy=True)

    await endpoint.send(PartStartEvent(index=0, part=TextPart(content="comp")))
    await endpoint.send(
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="leted"))
    )
    await endpoint.send(PartEndEvent(index=0, part=completed))
    await endpoint.send(boundary)
    await endpoint.send(PartStartEvent(index=0, part=pending))

    peer = IOEndpoint()
    endpoint.pair(peer)

    assert await peer.receive() == SnapshotEvent(None)
    first_start = await peer.receive()
    first_end = await peer.receive()
    assert isinstance(first_start, PartStartEvent)
    assert first_start.part == completed
    assert isinstance(first_end, PartEndEvent)
    assert first_end.part == completed
    assert await peer.receive() is boundary
    second_start = await peer.receive()
    assert isinstance(second_start, PartStartEvent)
    assert second_start.part == pending
