import asyncio
from dataclasses import dataclass

import pytest

from arox.core.io import ReplyEvent, RequestEvent, create_io_channel


@dataclass
class _Ping(RequestEvent):
    payload: str = ""


@dataclass
class _Pong(ReplyEvent):
    payload: str = ""


async def _drain(endpoint, recv_name, on_event):
    """Continuously receive on ``endpoint`` and forward non-reply events."""
    receive = getattr(endpoint, recv_name)
    while True:
        try:
            event = await receive()
        except Exception:
            return
        await on_event(event)


@pytest.mark.asyncio
async def test_request_reply_agent_to_adapter():
    agent_io, adapter_io = create_io_channel()
    async with agent_io, adapter_io:
        adapter_inbox: asyncio.Queue = asyncio.Queue()
        agent_inbox: asyncio.Queue = asyncio.Queue()
        adapter_drain = asyncio.create_task(
            _drain(adapter_io, "adapter_receive", adapter_inbox.put)
        )
        agent_drain = asyncio.create_task(
            _drain(agent_io, "agent_receive", agent_inbox.put)
        )

        async def adapter_handler():
            req = await adapter_inbox.get()
            assert isinstance(req, _Ping)
            assert req.payload == "hi"
            await adapter_io.adapter_send(_Pong(req_id=req.req_id, payload="ok"))

        handler_task = asyncio.create_task(adapter_handler())
        reply = await agent_io.agent_send(_Ping(payload="hi"))
        await handler_task

        adapter_drain.cancel()
        agent_drain.cancel()
        await asyncio.gather(adapter_drain, agent_drain, return_exceptions=True)

    assert isinstance(reply, _Pong)
    assert reply.payload == "ok"
    assert not agent_io._pending


@pytest.mark.asyncio
async def test_request_reply_adapter_to_agent():
    agent_io, adapter_io = create_io_channel()
    async with agent_io, adapter_io:
        adapter_inbox: asyncio.Queue = asyncio.Queue()
        agent_inbox: asyncio.Queue = asyncio.Queue()
        adapter_drain = asyncio.create_task(
            _drain(adapter_io, "adapter_receive", adapter_inbox.put)
        )
        agent_drain = asyncio.create_task(
            _drain(agent_io, "agent_receive", agent_inbox.put)
        )

        async def agent_handler():
            req = await agent_inbox.get()
            assert isinstance(req, _Ping)
            await agent_io.agent_send(_Pong(req_id=req.req_id, payload="from-agent"))

        handler_task = asyncio.create_task(agent_handler())
        reply = await adapter_io.adapter_send(_Ping(payload="hi"))
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

    agent_io, adapter_io = create_io_channel()
    async with agent_io, adapter_io:
        result = await adapter_io.adapter_send(Plain(value=42))
        assert result is None
        received = await asyncio.wait_for(agent_io.agent_receive(), timeout=1.0)
        assert isinstance(received, Plain)
        assert received.value == 42


@pytest.mark.asyncio
async def test_unknown_reply_is_dropped():
    @dataclass
    class Plain:
        pass

    agent_io, adapter_io = create_io_channel()
    async with agent_io, adapter_io:
        await adapter_io.adapter_send(_Pong(req_id="ghost", payload=""))
        await adapter_io.adapter_send(Plain())
        received = await asyncio.wait_for(agent_io.agent_receive(), timeout=1.0)
        assert isinstance(received, Plain)


@pytest.mark.asyncio
async def test_send_cancelled_clears_pending():
    agent_io, adapter_io = create_io_channel()
    async with agent_io, adapter_io:
        ping = _Ping(payload="x")
        send_task = asyncio.create_task(agent_io.agent_send(ping))
        await asyncio.sleep(0)
        assert ping.req_id in agent_io._pending

        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert ping.req_id not in agent_io._pending
