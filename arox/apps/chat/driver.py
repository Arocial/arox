import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentRunResult

from arox.core.agent_runtime import AgentRuntime
from arox.core.io import ReplyEvent, RequestEvent
from arox.core.plugin import CommandDispatchResult
from arox.core.runner import ServeRunner, cancel_task
from arox.core.types import UserInput

logger = logging.getLogger(__name__)


@dataclass
class ChatInputRequest(RequestEvent):
    pass


@dataclass
class ChatInputReply(UserInput, ReplyEvent):
    pass


async def dispatch_command(
    runtime: AgentRuntime, command: str | dict[str, Any]
) -> CommandDispatchResult:
    """Dispatch a command and render its outcome through the runtime channel."""
    result = await runtime.command_manager.dispatch(command)
    if result.status == "handled":
        output = result.reply.output if result.reply is not None else None
    elif result.status == "unknown":
        output = "Unknown command."
    elif result.status == "invalid":
        output = "Invalid command."
    else:
        output = None

    if output:
        await runtime.agent_ep.send(output)
    return result


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

            if reply.input_content is None:
                break

            text_input = reply.text_content
            if text_input and text_input.startswith("/"):
                await dispatch_command(runtime, text_input)
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

    async def cancel_current_execution(self) -> bool:
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
