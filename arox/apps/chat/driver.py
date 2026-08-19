import asyncio
import logging

from pydantic_ai import AgentRunResult

from arox.apps.chat.events import ChatInputReply, ChatInputRequest, StepDoneEvent
from arox.core.agent_runtime import AgentRuntime
from arox.core.runner import ServeRunner, cancel_task
from arox.core.types import UserInput

logger = logging.getLogger(__name__)


class ChatServeDriver:
    def __init__(self) -> None:
        self._interaction_task: asyncio.Task[AgentRunResult[str]] | None = None

    async def run(self, runner: ServeRunner) -> None:
        """Serve the runtime with an optional input generator."""
        runtime = runner.runtime
        assert runtime is not None
        while True:
            input_request = ChatInputRequest()

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
                await self._run_interaction(runtime, reply)

            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                else:
                    logger.info("Step cancelled.")
            except Exception as error:
                logger.error("An error occurred.", exc_info=error)

            await runtime.agent_ep.send(StepDoneEvent())

    async def cancel_current_interaction(self) -> bool:
        return await cancel_task(self._interaction_task)

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
