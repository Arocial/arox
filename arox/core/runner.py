import asyncio
import logging
from types import TracebackType
from typing import Protocol, Self

from pydantic_ai import AgentRunResult

from arox.core.agent_runtime import AgentRuntime
from arox.core.config import ConfigLoader
from arox.core.io import AbstractIOAdapter
from arox.core.session import AgentSession
from arox.core.types import UserInput

logger = logging.getLogger(__name__)


class SessionRunner:
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
        self.runtime: AgentRuntime | None = None

    async def __aenter__(self) -> Self:
        await self.start_runtime()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._stop_execution()
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

    async def start_runtime(self) -> AgentRuntime:
        async with self.session._runner_lock:
            if self.session.runner is not None:
                if self.session.runner is self and self.runtime is not None:
                    return self.runtime
                raise RuntimeError("Session is already active.")

            runtime = AgentRuntime(
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

    async def stop_runtime(self) -> None:
        await self.__aexit__(None, None, None)

    async def _stop_execution(self) -> None:
        pass


class TaskRunner(SessionRunner):
    def __init__(
        self,
        session: AgentSession,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
    ) -> None:
        super().__init__(session, config_loader, io_adapter)
        self._task: asyncio.Task[AgentRunResult[str]] | None = None
        self.result: AgentRunResult[str] | None = None
        self.error: str | None = None

    @property
    def task(self) -> asyncio.Task[AgentRunResult[str]] | None:
        return self._task

    def run(
        self,
        user_input: UserInput | str | None = None,
        *,
        render_user_message: bool = False,
    ) -> asyncio.Task[AgentRunResult[str]]:
        if self.runtime is None:
            raise RuntimeError("Runtime must be started before calling run().")
        if self._task is not None and not self._task.done():
            raise RuntimeError("A task is already running.")

        self.result = None
        self.error = None

        async def execute() -> AgentRunResult[str]:
            assert self.runtime is not None
            try:
                result = await self.runtime.run_turn(
                    user_input, render_user_message=render_user_message
                )
            except asyncio.CancelledError:
                self.error = "Task interrupted."
                raise
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc!s}"
                raise
            self.result = result
            return result

        task = asyncio.create_task(execute(), name=f"agent-turn:{self.session.id}")
        self._task = task
        return task

    async def _stop_execution(self) -> None:
        task = self._task
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class ServeDriver(Protocol):
    async def run(self, runner: "ServeRunner") -> None: ...

    async def cancel_current_interaction(self) -> bool: ...


class ServeRunner(SessionRunner):
    def __init__(
        self,
        session: AgentSession,
        config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
        driver: ServeDriver,
    ) -> None:
        super().__init__(session, config_loader, io_adapter)
        self.driver = driver
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def run(self) -> asyncio.Task[None]:
        if self.runtime is None:
            raise RuntimeError("Runtime must be started before calling run().")
        if self._task is not None and not self._task.done():
            raise RuntimeError("Session is already serving.")

        async def execute() -> None:
            try:
                await self.driver.run(self)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.session.record_error_event(exc)
                raise

        task = asyncio.create_task(execute(), name=f"agent-serve:{self.session.id}")
        self._task = task

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

    async def cancel_current_interaction(self) -> bool:
        return await self.driver.cancel_current_interaction()

    async def _stop_execution(self) -> None:
        task = self._task
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
