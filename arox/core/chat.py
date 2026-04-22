import asyncio
import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import DeferredToolResults
from pydantic_ai.tools import DeferredToolRequests

from arox.core.llm_base import MainAgent
from arox.core.plugin import CommandManager

logger = logging.getLogger(__name__)


class StepDoneEvent:
    pass


class ChatInputEvent:
    @dataclass
    class DeferredToolInput:
        question: str
        answer: str | None = None

    @dataclass
    class NormalInput:
        request: bool
        user_input: str | None

    @dataclass
    class ExceptionInput:
        exception: BaseException | None = None
        retry: bool = False

    def __init__(self):
        self.deferred_tools = OrderedDict[str, self.DeferredToolInput]()
        self.normal_input = self.NormalInput(False, "")
        self.exception_input = self.ExceptionInput()

        loop = asyncio.get_running_loop()
        self.future = loop.create_future()

    def add_deferred_tool(self, question: str, key: str):
        self.deferred_tools[key] = self.DeferredToolInput(question)

    def get_deferred_tool_input(self, key):
        return self.deferred_tools[key].answer

    async def wait(self):
        await self.future

    def generate_request(self):
        return {
            "deferred_tools": {k: t.question for k, t in self.deferred_tools.items()},
            "normal_input": {"request": self.normal_input.request},
            "exception_input": {
                "exception": f"{type(self.exception_input.exception).__name__}: {self.exception_input.exception}"
                if self.exception_input.exception
                else None
            },
        }

    def set_reply(self, reply: dict):
        if "deferred_tools" in reply:
            for k, v in reply["deferred_tools"].items():
                if k in self.deferred_tools:
                    self.deferred_tools[k].answer = v
        if "exception_input" in reply:
            self.exception_input.retry = reply["exception_input"]["retry"]
        if "normal_input" in reply:
            self.normal_input.user_input = reply["normal_input"]["user_input"]

        self.future.set_result(True)


class ChatAgent(MainAgent):
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

    @contextlib.asynccontextmanager
    async def chat_round(self):
        assert self.current_chat_input_event is not None
        await self.current_chat_input_event.wait()
        ctx = {"abort": False}
        try:
            yield ctx
        finally:
            if not ctx["abort"]:
                await self.agent_io.agent_send(self.current_chat_input_event)
                await self.agent_io.agent_send(StepDoneEvent())

    async def add_tool_input_request(self, question, key):
        assert self.current_chat_input_event is not None
        self.current_chat_input_event.add_deferred_tool(question, key)

    async def get_tool_input_result(self, key):
        assert self.current_chat_input_event is not None
        await self.current_chat_input_event.wait()
        return self.current_chat_input_event.get_deferred_tool_input(key)

    async def run(self):
        """Start the agent with optional input generator"""
        deferred_requests: DeferredToolRequests | None = None
        self.current_chat_input_event = ChatInputEvent()
        self.current_chat_input_event.normal_input.request = True
        await self.agent_io.agent_send(self.current_chat_input_event)

        while True:
            async with self.chat_round() as ctx:
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

                if self.current_chat_input_event.normal_input.request:
                    user_input = self.current_chat_input_event.normal_input.user_input
                    if user_input is None:
                        ctx["abort"] = True
                        break
                else:
                    user_input = None

                skip = (
                    self.current_chat_input_event.exception_input.exception
                    and not self.current_chat_input_event.exception_input.retry
                )

                self.current_chat_input_event = ChatInputEvent()
                if skip:
                    self.current_chat_input_event.normal_input.request = True
                    continue

                try:
                    if user_input is not None:
                        self.agent_session.add_event("user_input", {"text": user_input})
                        if not user_input.strip():
                            self.current_chat_input_event.normal_input.request = True
                            continue
                        is_command = await self.command_manager.try_execute_command(
                            user_input
                        )
                        if is_command:
                            self.agent_session.add_event(
                                "command", {"command": user_input}
                            )
                            self.current_chat_input_event.normal_input.request = True
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
                            self.current_chat_input_event.normal_input.request = True
                    except asyncio.CancelledError:
                        logger.info("Step cancelled.")
                        await self.agent_io.agent_send("\n[Step cancelled]\n")
                        deferred_requests = None
                        self.current_chat_input_event.normal_input.request = True
                    finally:
                        self.foreground_task = None

                except Exception as e:
                    logger.exception("An error occurred.")
                    self.agent_session.add_event(
                        "error", {"error": f"{type(e).__name__}: {e!s}"}
                    )
                    self.current_chat_input_event.exception_input.exception = e
