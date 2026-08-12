import asyncio
import inspect
import logging
import signal
from collections.abc import Awaitable, Callable
from types import FrameType, TracebackType
from typing import Any, Self

logger = logging.getLogger(__name__)


class SignalInterruptHandler:
    def __init__(
        self,
        callback: Callable[[], Awaitable[object] | object],
    ) -> None:
        self._callback = callback
        self._loop: asyncio.AbstractEventLoop | None = None
        self._original_handler: Any = None
        self._task: asyncio.Task[object] | None = None

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
        self._loop = None

        task = self._task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        logger.info("Received SIGINT, cancelling current step...")
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._invoke_callback)

    def _invoke_callback(self) -> None:
        if self._task is not None and not self._task.done():
            return

        try:
            result = self._callback()
        except Exception:
            logger.exception("Interrupt handler failed")
            return

        if inspect.isawaitable(result):
            self._task = asyncio.create_task(self._await_callback(result))

    async def _await_callback(self, result: Awaitable[Any]) -> object:
        try:
            return await result
        except Exception:
            logger.exception("Async interrupt handler failed")
            raise
