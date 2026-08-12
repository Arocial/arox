import asyncio
import signal
from collections.abc import Callable
from typing import Any, cast

import pytest

from arox.apps.chat.interrupt import SignalInterruptHandler


@pytest.mark.asyncio
async def test_signal_interrupt_handler_invokes_sync_callback():
    calls = 0

    def callback() -> None:
        nonlocal calls
        calls += 1

    async with SignalInterruptHandler(callback):
        handler = cast(Callable[[int, Any], Any], signal.getsignal(signal.SIGINT))
        handler(signal.SIGINT, None)
        await asyncio.sleep(0)

    assert calls == 1


@pytest.mark.asyncio
async def test_signal_interrupt_handler_deduplicates_async_callback():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def callback() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()

    async with SignalInterruptHandler(callback):
        handler = cast(Callable[[int, Any], Any], signal.getsignal(signal.SIGINT))
        handler(signal.SIGINT, None)
        await asyncio.wait_for(started.wait(), timeout=1)
        handler(signal.SIGINT, None)
        await asyncio.sleep(0)
        assert calls == 1
        release.set()


@pytest.mark.asyncio
async def test_signal_interrupt_handler_restores_original_handler():
    original_handler = signal.getsignal(signal.SIGINT)

    async with SignalInterruptHandler(lambda: None):
        assert signal.getsignal(signal.SIGINT) != original_handler

    assert signal.getsignal(signal.SIGINT) == original_handler
