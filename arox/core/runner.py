import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic_ai import AgentRunResult

from arox.core.config import ConfigLoader
from arox.core.io import AbstractIOAdapter
from arox.core.llm_base import LLMBaseAgent
from arox.core.session import AgentSession
from arox.core.types import UserInput

logger = logging.getLogger(__name__)


class SessionRunner(ABC):
    """Own the ephemeral execution state attached to an AgentSession."""

    def __init__(
        self,
        session: AgentSession,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
    ) -> None:
        self.session = session
        self.config_loader = config_loader
        self.io_adapter = io_adapter
        self.runtime: LLMBaseAgent | None = None

    async def start(self) -> LLMBaseAgent:
        async with self.session._runner_lock:
            if self.session.runner is not None:
                if self.session.runner is self and self.runtime is not None:
                    return self.runtime
                raise RuntimeError("Session is already active.")

            runtime = LLMBaseAgent(
                parent_config_loader=self.config_loader,
                io_adapter=self.io_adapter,
                session=self.session,
            )
            self.session.runner = self
            self.runtime = runtime
            try:
                await runtime.__aenter__()
            except BaseException:
                try:
                    await runtime.__aexit__(None, None, None)
                finally:
                    self.runtime = None
                    self.session.runner = None
                raise
            if self.session.manager:
                self.session.manager._track(self.session, self.session.owner)
            return runtime

    async def _stop_runtime(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> None:
        async with self.session._runner_lock:
            if self.session.runner is not self:
                return
            runtime = self.runtime
            try:
                if runtime is not None:
                    await runtime.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                self.runtime = None
                self.session.runner = None

    @abstractmethod
    async def stop(self) -> None:
        """Stop all execution and release the runtime."""


class TaskRunner(SessionRunner):
    def __init__(
        self,
        session: AgentSession,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
    ) -> None:
        super().__init__(session, config_loader, io_adapter)
        self._task: asyncio.Task[AgentRunResult[str]] | None = None

    @property
    def current_task(self) -> asyncio.Task[AgentRunResult[str]] | None:
        return self._task

    def run(
        self, user_input: UserInput | str | None = None
    ) -> asyncio.Task[AgentRunResult[str]]:
        if self.runtime is None:
            raise RuntimeError("Runner must be started before calling run().")
        if self._task is not None and not self._task.done():
            raise RuntimeError("Session is already running.")

        async def execute() -> AgentRunResult[str]:
            try:
                assert self.runtime is not None
                return await self.runtime.step(user_input)
            finally:
                if self._task is task:
                    self._task = None

        task = asyncio.create_task(execute(), name=f"agent-turn:{self.session.id}")
        self._task = task
        return task

    async def wait(self, timeout: float | None = None) -> AgentRunResult[str] | None:
        task = self._task
        if task is None:
            return None
        if timeout is None:
            return await asyncio.shield(task)
        return await asyncio.wait_for(asyncio.shield(task), timeout)

    async def cancel(self) -> bool:
        task = self._task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def stop(self) -> None:
        await self.cancel()
        await self._stop_runtime()


class ServeDriver(Protocol):
    async def serve(self, runner: "ServingRunner") -> None: ...


class ServingRunner(SessionRunner):
    def __init__(
        self,
        session: AgentSession,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
        driver: ServeDriver,
    ) -> None:
        super().__init__(session, config_loader, io_adapter)
        self.driver = driver
        self._serve_task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[AgentRunResult[str]] | None = None

    @property
    def serve_task(self) -> asyncio.Task[None] | None:
        return self._serve_task

    @property
    def current_task(self) -> asyncio.Task[AgentRunResult[str]] | None:
        return self._turn_task

    async def start_serving(self) -> asyncio.Task[None]:
        await self.start()
        return self.serve()

    def serve(self) -> asyncio.Task[None]:
        if self.runtime is None:
            raise RuntimeError("Runner must be started before calling serve().")
        if self._serve_task is not None and not self._serve_task.done():
            raise RuntimeError("Session is already serving.")

        async def execute() -> None:
            try:
                await self.driver.serve(self)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.session.record_error_event(exc)
                raise
            finally:
                try:
                    await self._stop_runtime()
                finally:
                    if self._serve_task is task:
                        self._serve_task = None

        task = asyncio.create_task(execute(), name=f"agent-serve:{self.session.id}")
        self._serve_task = task

        def log_failure(completed: asyncio.Task[None]) -> None:
            if completed.cancelled():
                return
            if exc := completed.exception():
                logger.error(
                    "Session %s serve loop failed: %s",
                    self.session.id,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(log_failure)
        return task

    def run_turn(
        self, user_input: UserInput | str | None = None
    ) -> asyncio.Task[AgentRunResult[str]]:
        if self.runtime is None:
            raise RuntimeError("Runner must be started before running a turn.")
        if self._turn_task is not None and not self._turn_task.done():
            raise RuntimeError("A turn is already running.")

        async def execute() -> AgentRunResult[str]:
            try:
                assert self.runtime is not None
                return await self.runtime.step(user_input)
            finally:
                if self._turn_task is task:
                    self._turn_task = None

        task = asyncio.create_task(execute(), name=f"agent-turn:{self.session.id}")
        self._turn_task = task
        return task

    async def cancel_turn(self) -> bool:
        task = self._turn_task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def wait(self) -> None:
        task = self._serve_task
        if task is not None:
            await asyncio.shield(task)

    async def stop(self) -> None:
        await self.cancel_turn()
        task = self._serve_task
        if task is not None and task is not asyncio.current_task():
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._stop_runtime()
