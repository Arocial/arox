import asyncio
import logging
from dataclasses import dataclass

from arox.core.io import ReplyEvent, RequestEvent
from arox.core.llm_base import UserInput
from arox.core.runner import ServingRunner

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
    async def serve(self, runner: ServingRunner) -> None:
        """Start the agent with optional input generator"""
        runtime = runner.runtime
        assert runtime is not None
        pending_exception: BaseException | None = None

        while True:
            # 1. Prepare the event for this round
            input_request = ChatInputRequest()

            if pending_exception:
                input_request.pending_exception = pending_exception
                pending_exception = None

            # 2. Send the request and wait for the matching reply
            reply: ChatInputReply = await runtime.agent_io.send(input_request)

            if reply.is_abort(input_request):
                break

            # 5. Execute the step
            try:
                result = await runner.run_turn(reply)

                if result and isinstance(result.output, Exception):
                    e = result.output
                    logger.error("An error occurred.", exc_info=e)
                    runner.session.record_turn_error(e)
                    pending_exception = e

            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                logger.info("Step cancelled.")
                await runtime.agent_io.send("\n[Step cancelled]\n")

            # 6. Send StepDoneEvent to indicate the step is finished
            await runtime.agent_io.send(StepDoneEvent())
