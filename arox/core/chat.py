import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import DeferredToolResults
from pydantic_ai.tools import DeferredToolRequests

from arox.core.io import ReplyEvent, RequestEvent
from arox.core.llm_base import DelegatableAgent, MainAgent, UserInput
from arox.core.session import AgentSession
from arox.plugins.slots import AGENT_ERROR

logger = logging.getLogger(__name__)


class StepDoneEvent:
    pass


@dataclass
class _DeferredToolQuestion:
    question: str


@dataclass
class _NormalInputRequest:
    request: bool = False


@dataclass
class _ExceptionInputRequest:
    exception: BaseException | None = None


@dataclass
class ChatInputEvent(RequestEvent):
    DeferredToolInput = _DeferredToolQuestion
    NormalInput = _NormalInputRequest
    ExceptionInput = _ExceptionInputRequest

    deferred_tools: OrderedDict[str, _DeferredToolQuestion] = field(
        default_factory=OrderedDict
    )
    normal_input: _NormalInputRequest = field(default_factory=_NormalInputRequest)
    exception_input: _ExceptionInputRequest = field(
        default_factory=_ExceptionInputRequest
    )

    def add_deferred_tool(self, question: str, key: str) -> None:
        self.deferred_tools[key] = _DeferredToolQuestion(question)

    def generate_request(self) -> dict:
        return {
            "req_id": self.req_id,
            "deferred_tools": {k: t.question for k, t in self.deferred_tools.items()},
            "normal_input": {"request": self.normal_input.request},
            "exception_input": {
                "exception": f"{type(self.exception_input.exception).__name__}: {self.exception_input.exception}"
                if self.exception_input.exception
                else None
            },
        }


@dataclass
class ChatInputReply(UserInput, ReplyEvent):
    deferred_answers: dict[str, str | None] = field(default_factory=dict)
    retry: bool = False

    def is_abort(self, request: ChatInputEvent) -> bool:
        for key, tool in request.deferred_tools.items():
            if tool.question and self.deferred_answers.get(key) is None:
                return True
        return bool(request.normal_input.request and self.user_input is None)

    def is_skip(self, request: ChatInputEvent) -> bool:
        return request.exception_input.exception is not None and not self.retry


class ChatAgent(MainAgent, DelegatableAgent):
    def __init__(
        self,
        parsed_config,
        io_adapter,
        session: AgentSession,
        workspace: Path | str | None = None,
    ):
        self.foreground_task: asyncio.Task | None = None
        super().__init__(
            parsed_config,
            io_adapter,
            session,
            workspace,
        )

    def cancel_foreground_task(self):
        if self.foreground_task:
            self.foreground_task.cancel()

    async def run(self):
        """Start the agent with optional input generator"""
        deferred_requests: DeferredToolRequests | None = None
        pending_exception: BaseException | None = None

        while True:
            # 1. Prepare the event for this round
            event = ChatInputEvent()
            event.normal_input.request = True

            if pending_exception:
                event.exception_input.exception = pending_exception
                pending_exception = None

            # 2. Send the request and wait for the matching reply
            reply: ChatInputReply = await self.agent_io.send(event)

            if reply.is_abort(event):
                break

            if reply.is_skip(event):
                await self.agent_io.send(StepDoneEvent())
                continue

            # 5. Execute the step
            try:
                if deferred_requests:
                    deferred_results = DeferredToolResults()
                    for call in deferred_requests.calls:
                        deferred_results.calls[
                            call.tool_call_id
                        ] = await deferred_requests.metadata[call.tool_call_id][
                            "result_callback"
                        ]()
                else:
                    deferred_results = None

                step_task = asyncio.create_task(
                    self.step(reply, deferred_tool_results=deferred_results)
                )
                self.foreground_task = step_task
                try:
                    result = await step_task
                    if result and isinstance(result.output, DeferredToolRequests):
                        deferred_requests = result.output
                    else:
                        deferred_requests = None
                except asyncio.CancelledError:
                    logger.info("Step cancelled.")
                    await self.agent_io.send("\n[Step cancelled]\n")
                    deferred_requests = None
                finally:
                    self.foreground_task = None

            except Exception as e:
                logger.exception("An error occurred.")
                await self.invoke_slot(AGENT_ERROR, e)
                self.session.record_error(e)
                pending_exception = e

            # 6. Send StepDoneEvent to indicate the step is finished
            await self.agent_io.send(StepDoneEvent())
