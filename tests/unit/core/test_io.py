import asyncio
from dataclasses import dataclass

import pytest

from arox.core.io import (
    AbstractIOAdapter,
    IOHost,
    ReplyEvent,
    RequestEvent,
)


@dataclass
class _Ping(RequestEvent):
    payload: str = ""


@dataclass
class _Pong(ReplyEvent):
    payload: str = ""


class _RecordingAdapter(AbstractIOAdapter):
    def __init__(self):
        super().__init__()
        self.events: asyncio.Queue = asyncio.Queue()
        self.closed_endpoints = []

    async def handle_event(self, adapter_io, event):
        await self.events.put((adapter_io, event))

    async def on_endpoint_closed(self, adapter_io):
        self.closed_endpoints.append(adapter_io)


async def _drain(endpoint, on_event):
    """Continuously receive on ``endpoint`` and forward non-reply events."""
    while True:
        try:
            event = await endpoint.receive()
        except Exception:
            return
        await on_event(event)


def _create_test_channel():
    host = IOHost(_RecordingAdapter())
    return host.agent_io, host.adapter_io


@pytest.mark.asyncio
async def test_request_reply_agent_to_adapter():
    agent_io, adapter_io = _create_test_channel()
    async with agent_io, adapter_io:
        adapter_inbox: asyncio.Queue = asyncio.Queue()
        agent_inbox: asyncio.Queue = asyncio.Queue()
        adapter_drain = asyncio.create_task(_drain(adapter_io, adapter_inbox.put))
        agent_drain = asyncio.create_task(_drain(agent_io, agent_inbox.put))

        async def adapter_handler():
            req = await adapter_inbox.get()
            assert isinstance(req, _Ping)
            assert req.payload == "hi"
            await adapter_io.send(_Pong(req_id=req.req_id, payload="ok"))

        handler_task = asyncio.create_task(adapter_handler())
        reply = await agent_io.send(_Ping(payload="hi"))
        await handler_task

        adapter_drain.cancel()
        agent_drain.cancel()
        await asyncio.gather(adapter_drain, agent_drain, return_exceptions=True)

    assert isinstance(reply, _Pong)
    assert reply.payload == "ok"
    assert not agent_io._pending


@pytest.mark.asyncio
async def test_request_reply_adapter_to_agent():
    agent_io, adapter_io = _create_test_channel()
    async with agent_io, adapter_io:
        adapter_inbox: asyncio.Queue = asyncio.Queue()
        agent_inbox: asyncio.Queue = asyncio.Queue()
        adapter_drain = asyncio.create_task(_drain(adapter_io, adapter_inbox.put))
        agent_drain = asyncio.create_task(_drain(agent_io, agent_inbox.put))

        async def agent_handler():
            req = await agent_inbox.get()
            assert isinstance(req, _Ping)
            await agent_io.send(_Pong(req_id=req.req_id, payload="from-agent"))

        handler_task = asyncio.create_task(agent_handler())
        reply = await adapter_io.send(_Ping(payload="hi"))
        await handler_task

        adapter_drain.cancel()
        agent_drain.cancel()
        await asyncio.gather(adapter_drain, agent_drain, return_exceptions=True)

    assert isinstance(reply, _Pong)
    assert reply.payload == "from-agent"
    assert not adapter_io._pending


@pytest.mark.asyncio
async def test_fire_and_forget_send_returns_none():
    @dataclass
    class Plain:
        value: int = 0

    agent_io, adapter_io = _create_test_channel()
    async with agent_io, adapter_io:
        result = await adapter_io.send(Plain(value=42))
        assert result is None
        received = await asyncio.wait_for(agent_io.receive(), timeout=1.0)
        assert isinstance(received, Plain)
        assert received.value == 42


@pytest.mark.asyncio
async def test_unknown_reply_is_dropped():
    @dataclass
    class Plain:
        pass

    agent_io, adapter_io = _create_test_channel()
    async with agent_io, adapter_io:
        await adapter_io.send(_Pong(req_id="ghost", payload=""))
        await adapter_io.send(Plain())
        received = await asyncio.wait_for(agent_io.receive(), timeout=1.0)
        assert isinstance(received, Plain)


@pytest.mark.asyncio
async def test_send_cancelled_clears_pending():
    agent_io, adapter_io = _create_test_channel()
    async with agent_io, adapter_io:
        ping = _Ping(payload="x")
        send_task = asyncio.create_task(agent_io.send(ping))
        await asyncio.sleep(0)
        assert ping.req_id in agent_io._pending

        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert ping.req_id not in agent_io._pending


@pytest.mark.asyncio
async def test_io_host_owns_adapter_event_loop_and_cleanup():
    adapter = _RecordingAdapter()
    host = IOHost(adapter)

    assert host.agent_io.host is host
    assert host.adapter_io.host is host
    assert host.uuid not in adapter.hosts

    async with host:
        assert adapter.hosts[host.uuid] is host
        await host.agent_io.send("hello")
        endpoint, start_event = await asyncio.wait_for(adapter.events.get(), timeout=1)
        _, end_event = await asyncio.wait_for(adapter.events.get(), timeout=1)

        assert endpoint is host.adapter_io
        assert start_event.part.content == "hello"
        assert end_event.part.content == "hello"

    assert adapter.closed_endpoints == [host.adapter_io]
    assert host.uuid not in adapter.hosts
