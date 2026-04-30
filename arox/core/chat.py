import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import DeferredToolResults
from pydantic_ai.tools import DeferredToolRequests

from arox.core.io import ReplyEvent, RequestEvent
from arox.core.llm_base import DelegatableAgent, MainAgent
from arox.core.plugin import CommandManager

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
            "deferred_tools": {k: t.question for k, t in self.deferred_tools.items()},
            "normal_input": {"request": self.normal_input.request},
            "exception_input": {
                "exception": f"{type(self.exception_input.exception).__name__}: {self.exception_input.exception}"
                if self.exception_input.exception
                else None
            },
        }


@dataclass
class ChatInputReply(ReplyEvent):
    deferred_answers: dict[str, str | None] = field(default_factory=dict)
    user_input: str | None = None
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
        name,
        parsed_config,
        io_adapter,
        local_toolset=None,
        workspace: Path | str | None = None,
    ):
        self.command_manager = CommandManager(self)
        self.foreground_task: asyncio.Task | None = None
        self.current_chat_input_event: ChatInputEvent | None = None
        super().__init__(
            name,
            parsed_config,
            io_adapter,
            local_toolset,
            workspace,
        )

    def cancel_foreground_task(self):
        if self.foreground_task:
            self.foreground_task.cancel()

    def load_plugins(self):
        plugins = super().load_plugins()
        for plugin in plugins:
            # Register commands
            cmds = plugin.commands()
            if cmds:
                self.command_manager.register_commands(cmds)
        return plugins

    async def run(self):
        """Start the agent with optional input generator"""
        deferred_requests: DeferredToolRequests | None = None
        pending_exception: BaseException | None = None

        while True:
            # 1. Prepare the event for this round
            event = ChatInputEvent()
            event.normal_input.request = True
            self.current_chat_input_event = event

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

                user_input = reply.user_input
                if user_input is not None:
                    self.agent_session.add_event("user_input", {"text": user_input})
                    if not user_input.strip():
                        await self.agent_io.send(StepDoneEvent())
                        continue
                    is_command = await self.command_manager.try_execute_command(
                        user_input
                    )
                    if is_command:
                        self.agent_session.add_event("command", {"command": user_input})
                        await self.agent_io.send(StepDoneEvent())
                        continue

                step_task = asyncio.create_task(
                    self.step(user_input, deferred_tool_results=deferred_results)
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
                self.agent_session.add_event(
                    "error", {"error": f"{type(e).__name__}: {e!s}"}
                )
                pending_exception = e

            # 6. Send StepDoneEvent to indicate the step is finished
            await self.agent_io.send(StepDoneEvent())
