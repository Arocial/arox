import asyncio
import logging
from dataclasses import dataclass

from pydantic_ai import AgentRunResult

from arox.core.agent_runtime import AgentRuntime
from arox.core.io import ReplyEvent, RequestEvent
from arox.core.runner import ServeRunner
from arox.core.types import UserInput

logger = logging.getLogger(__name__)


class StepDoneEvent:
    pass


@dataclass
class ChatInputRequest(RequestEvent):
    request_normal_input: bool = True
    pending_exception: BaseException | None = None


@dataclass
class ChatInputReply(UserInput, ReplyEvent):
    def is_abort(self, request: ChatInputRequest) -> bool:
        return bool(request.request_normal_input and self.input_content is None)


class ChatServeDriver:
    def __init__(self) -> None:
        self._interaction_task: asyncio.Task[AgentRunResult[str]] | None = None

    async def run(self, runner: ServeRunner) -> None:
        """Serve the runtime with an optional input generator."""
        runtime = runner.runtime
        assert runtime is not None
        pending_exception: BaseException | None = None

        while True:
            input_request = ChatInputRequest()

            if pending_exception:
                input_request.pending_exception = pending_exception
                pending_exception = None

            reply: ChatInputReply = await runtime.agent_ep.send(input_request)

            if reply.is_abort(input_request):
                break

            text_input = reply.text_content
            if text_input and text_input.startswith("/"):
                command_reply = await runtime.command_manager.try_handle_slash(
                    text_input
                )
                if command_reply is not None:
                    if command_reply.output:
                        await runtime.agent_ep.send(command_reply.output)
                    continue

            try:
                result = await self._run_interaction(runtime, reply)

                if result and isinstance(result.output, BaseException):
                    error = result.output
                    logger.error("An error occurred.", exc_info=error)
                    pending_exception = error

            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                logger.info("Step cancelled.")
                await runtime.agent_ep.send("\n[Step cancelled]\n")

            await runtime.agent_ep.send(StepDoneEvent())

    async def cancel_current_interaction(self) -> bool:
        task = self._interaction_task
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def _run_interaction(
        self,
        runtime: AgentRuntime,
        user_input: UserInput | str | None,
    ) -> AgentRunResult[str]:
        if self._interaction_task is not None and not self._interaction_task.done():
            raise RuntimeError("An interaction is already running.")

        task = asyncio.create_task(
            runtime.run_turn(user_input),
            name=f"agent-interaction:{runtime.session.id}",
        )
        self._interaction_task = task
        try:
            return await task
        finally:
            if self._interaction_task is task:
                self._interaction_task = None
