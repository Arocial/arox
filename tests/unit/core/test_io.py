import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pydantic_ai import PartEndEvent, PartStartEvent, TextPart
from pydantic_ai.messages import ModelResponse

from arox.core.io import AbstractIOAdapter, AgentIOEndpoint, IOEndpoint, SnapshotEvent
from arox.core.session import AgentSession
from arox.core.types import (
    ClientInput,
    CommandPayload,
    TurnStateEvent,
    normalize_client_input,
)

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

    agent_ep = AgentIOEndpoint(AgentSession(agent_name="test"))
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
    agent_ep = AgentIOEndpoint(AgentSession(agent_name="test"))
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
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session)
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
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session, replay_threshold=1)
    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(None)

    session.record_error_event("failed")
    await endpoint.send(TurnStateEvent(busy=False))
    assert await peer.receive() == TurnStateEvent(busy=False)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(peer.receive(), timeout=0.01)

    endpoint.disconnect()
    replacement = IOEndpoint()
    endpoint.pair(replacement)

    assert await replacement.receive() == SnapshotEvent(session.journal[-1].id)


@pytest.mark.asyncio
async def test_adapter_connect_replaces_old_peer():
    adapter = _RecordingAdapter()
    agent_ep = AgentIOEndpoint(AgentSession(agent_name="test"))
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
async def test_replay_threshold_waits_for_turn_and_concurrent_command_to_finish():
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session, replay_threshold=1)
    await endpoint.send(TurnStateEvent(busy=True))
    part = TextPart(content="streamed answer")
    await endpoint.send(PartStartEvent(index=0, part=part))
    command = normalize_client_input(
        ClientInput(payload=CommandPayload(command="/info", status="accepted"))
    )
    await endpoint.send(command)
    session.record_model_message(ModelResponse(parts=[part]), run_id="run", sequence=0)
    await endpoint.send(PartEndEvent(index=0, part=part))
    await endpoint.send(TurnStateEvent(busy=False))
    assert endpoint._snapshot_journal_id is None
    assert len(endpoint._cached_events) > endpoint._replay_threshold

    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(None)
    assert await peer.receive() == TurnStateEvent(busy=True)
    assert isinstance(await peer.receive(), PartStartEvent)
    assert await peer.receive() is command
    endpoint.disconnect()
    completed = session.record_command_completed(command, "handled", output="details")
    await endpoint.send(completed)
    assert endpoint._snapshot_journal_id == completed.id
    assert endpoint._cached_events == [TurnStateEvent(busy=False)]
    replacement = IOEndpoint()
    endpoint.pair(replacement)
    assert await replacement.receive() == SnapshotEvent(completed.id)
    assert await replacement.receive() == TurnStateEvent(busy=False)


@pytest.mark.asyncio
async def test_command_completion_during_stream_does_not_advance_snapshot():
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session, replay_threshold=1)
    await endpoint.send(TurnStateEvent(busy=True))
    part = TextPart(content="partial")
    await endpoint.send(PartStartEvent(index=0, part=part))
    command = normalize_client_input(
        ClientInput(payload=CommandPayload(command="/info", status="accepted"))
    )
    await endpoint.send(command)
    completed = session.record_command_completed(command, "handled", output="details")
    await endpoint.send(completed)
    assert endpoint._snapshot_journal_id is None
    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(None)
    assert await peer.receive() == TurnStateEvent(busy=True)
    assert isinstance(await peer.receive(), PartStartEvent)
    assert await peer.receive() is command
    assert await peer.receive() is completed


@pytest.mark.asyncio
async def test_threshold_compacts_completed_prefix_and_retains_new_stream_tail():
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session, replay_threshold=5)
    connected = IOEndpoint()
    endpoint.pair(connected)
    assert await connected.receive() == SnapshotEvent(None)
    part = TextPart(content="completed")
    await endpoint.send(TurnStateEvent(busy=True))
    await endpoint.send(PartStartEvent(index=0, part=part))
    await endpoint.send(PartEndEvent(index=0, part=part))
    committed = session.record_model_message(
        ModelResponse(parts=[part]), run_id="run", sequence=0
    )
    await endpoint.send(TurnStateEvent(busy=False))
    assert endpoint._snapshot_journal_id is None
    await endpoint.send(TurnStateEvent(busy=True))
    assert endpoint._snapshot_journal_id == committed.id
    # Compaction changes future replay, never an already connected consumer's inbox.
    assert await connected.receive() == TurnStateEvent(busy=True)
    assert isinstance(await connected.receive(), PartStartEvent)
    assert isinstance(await connected.receive(), PartEndEvent)
    assert await connected.receive() == TurnStateEvent(busy=False)
    assert await connected.receive() == TurnStateEvent(busy=True)
    endpoint.disconnect()
    pending = PartStartEvent(index=0, part=TextPart(content="new partial"))
    await endpoint.send(pending)
    # A later journal append must not change the boundary of the replay prefix.
    session.record_error_event("later entry")
    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(committed.id)
    assert await peer.receive() == TurnStateEvent(busy=False)
    assert await peer.receive() == TurnStateEvent(busy=True)
    assert await peer.receive() is pending
    assert len(session.build_io_timeline(through_id=committed.id)) == 1


@pytest.mark.asyncio
async def test_snapshot_preserves_unjournaled_output_and_latest_state():
    session = AgentSession(agent_name="test")
    endpoint = AgentIOEndpoint(session, replay_threshold=1)
    await endpoint.send(TurnStateEvent(busy=True))
    await endpoint.send("temporary notice")
    session.record_error_event("failed")
    await endpoint.send(TurnStateEvent(busy=False))
    peer = IOEndpoint()
    endpoint.pair(peer)
    assert await peer.receive() == SnapshotEvent(session.journal[-1].id)
    assert (await peer.receive()).part.content == "temporary notice"
    assert (await peer.receive()).part.content == "temporary notice"
    assert await peer.receive() == TurnStateEvent(busy=False)
