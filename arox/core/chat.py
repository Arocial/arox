import asyncio
import logging
from dataclasses import dataclass

from arox.core.io import ReplyEvent, RequestEvent
from arox.core.llm_base import DelegatableAgent, MainAgent, UserInput
from arox.core.session import AgentSession

logger = logging.getLogger(__name__)


class StepDoneEvent:
    pass


@dataclass
class ChatInputRequest(RequestEvent):
    request_normal_input: bool = True
    pending_exception: BaseException | None = None

    def generate_request(self) -> dict:
        return {
            "req_id": self.req_id,
            "normal_input": {"request": self.request_normal_input},
            "exception_input": {
                "exception": f"{type(self.pending_exception).__name__}: {self.pending_exception}"
                if self.pending_exception
                else None
            },
        }


@dataclass
class ChatInputReply(UserInput, ReplyEvent):
    def is_abort(self, request: ChatInputRequest) -> bool:
        return bool(request.request_normal_input and self.user_input is None)


class ChatAgent(MainAgent, DelegatableAgent):
    def __init__(
        self,
        parsed_config,
        io_adapter,
        session: AgentSession,
    ):
        self.foreground_task: asyncio.Task | None = None
        super().__init__(
            parsed_config,
            io_adapter,
            session,
        )

    def cancel_foreground_task(self):
        if self.foreground_task:
            self.foreground_task.cancel()

    async def run(self):
        """Start the agent with optional input generator"""
        pending_exception: BaseException | None = None

        while True:
            # 1. Prepare the event for this round
            input_request = ChatInputRequest()

            if pending_exception:
                input_request.pending_exception = pending_exception
                pending_exception = None

            # 2. Send the request and wait for the matching reply
            reply: ChatInputReply = await self.agent_io.send(input_request)

            if reply.is_abort(input_request):
                break

            # 5. Execute the step
            try:
                step_task = asyncio.create_task(self.step(reply))
                self.foreground_task = step_task

                result = await step_task

                if result and isinstance(result.output, Exception):
                    e = result.output
                    logger.error("An error occurred.", exc_info=e)
                    self.session.record_error(e)
                    pending_exception = e

            except asyncio.CancelledError:
                logger.info("Step cancelled.")
                await self.agent_io.send("\n[Step cancelled]\n")
            finally:
                self.foreground_task = None

            # 6. Send StepDoneEvent to indicate the step is finished
            await self.agent_io.send(StepDoneEvent())
