import logging

from pydantic_ai.tools import DeferredToolRequests

from arox import commands
from arox.agent_patterns.llm_base import LLMBaseAgent
from arox.agent_patterns.state import SimpleState
from arox.commands.manager import CommandManager

logger = logging.getLogger(__name__)


class ChatAgent(LLMBaseAgent):
    def __init__(
        self,
        name,
        config_parser,
        agent_io,
        local_toolset=None,
        state_cls=SimpleState,
        context=None,
    ):
        if context is None:
            context = {}
        super().__init__(
            name,
            config_parser,
            agent_io,
            local_toolset,
            state_cls,
            context=context,
        )

        self.command_manager = CommandManager(self)

    def register_commands(self, cmds: list[commands.Command]):
        self.command_manager.register_commands(cmds)

    async def start(self):
        """Start the agent with optional input generator"""
        chat_mode = "normal"

        deferred_requests = None

        while True:
            async with self.agent_io.chat_round(deferred_requests) as (
                user_input,
                deferred_results,
            ):
                try:
                    if chat_mode == "normal":
                        result = None
                        if deferred_requests:
                            result = await self.agent_io.run_cancellable(
                                self.step(None, deferred_tool_results=deferred_results)
                            )
                            deferred_requests = None
                        else:
                            if isinstance(user_input, EOFError):
                                break
                            if not user_input.strip():
                                continue
                            is_command = await self.command_manager.try_execute_command(
                                user_input
                            )
                            if not is_command:
                                result = await self.agent_io.run_cancellable(
                                    self.step(user_input)
                                )
                        if result and isinstance(result.output, DeferredToolRequests):
                            deferred_requests = result.output

                    elif chat_mode == "ask_continue":
                        if (
                            isinstance(user_input, EOFError)
                            or user_input.strip().lower() != "y"
                        ):
                            break
                        chat_mode = "normal"
                except Exception as e:
                    logger.exception("An error occurred.")
                    await self.agent_io.agent_send(f"An error occurred: {e}")
                    await self.agent_io.agent_send("Do you want to continue? (y/n)")
                    chat_mode = "ask_continue"
