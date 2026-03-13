import logging

from pydantic_ai import DeferredToolResults
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
        deferred_requests: DeferredToolRequests | None = None
        chat_input_event = self.agent_io.create_chat_input_event()
        chat_input_event.normal_input.request = True
        await self.agent_io.agent_send(chat_input_event)

        while True:
            async with self.agent_io.chat_round():
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
                if (
                    chat_input_event.exception_input.exception
                    and chat_input_event.exception_input.to_continue
                ):
                    user_input = "Please continue."
                elif chat_input_event.normal_input.request:
                    user_input = chat_input_event.normal_input.user_input
                else:
                    user_input = None

                chat_input_event = self.agent_io.create_chat_input_event()

                if user_input is None and deferred_results is None:
                    break

                try:
                    if user_input is not None:
                        if not user_input.strip():
                            chat_input_event.normal_input.request = True
                            continue
                        is_command = await self.command_manager.try_execute_command(
                            user_input
                        )
                        if is_command:
                            chat_input_event.normal_input.request = True
                            continue

                    result = await self.agent_io.run_cancellable(
                        self.step(user_input, deferred_tool_results=deferred_results)
                    )
                    if result and isinstance(result.output, DeferredToolRequests):
                        deferred_requests = result.output
                    else:
                        deferred_requests = None
                        chat_input_event.normal_input.request = True

                except Exception as e:
                    logger.exception("An error occurred.")
                    chat_input_event.exception_input.exception = e
