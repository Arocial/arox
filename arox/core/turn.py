import asyncio
from typing import Any

from pydantic_ai import AgentRunResult

from arox.core.types import ClientInput


class Turn:
    """One execution of an agent runtime for a user input."""

    def __init__(
        self,
        user_input: ClientInput,
        task: asyncio.Task[AgentRunResult[str]],
    ) -> None:
        self.user_input = user_input
        self.task = task

    @property
    def done(self) -> bool:
        return self.task.done()

    @property
    def result(self) -> AgentRunResult[str] | None:
        if not self.task.done() or self.task.cancelled():
            return None
        if self.task.exception() is not None:
            return None
        return self.task.result()

    @property
    def error(self) -> BaseException | None:
        if not self.task.done():
            return None
        if self.task.cancelled():
            return asyncio.CancelledError()
        return self.task.exception()

    async def wait(self) -> AgentRunResult[str]:
        return await self.task

    async def cancel(self) -> bool:
        if self.task.done() or self.task is asyncio.current_task():
            return False
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        return True

    def __await__(self) -> Any:
        return self.wait().__await__()
