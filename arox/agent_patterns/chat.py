from arox import commands
from arox.agent_patterns.llm_base import LLMBaseAgent
from arox.agent_patterns.state import SimpleState
from arox.commands.manager import CommandManager


class ChatAgent(LLMBaseAgent):
    def __init__(
        self,
        name,
        config_parser,
        io_adapter,
        local_toolset=None,
        state_cls=SimpleState,
        context={},
    ):
        super().__init__(
            name,
            config_parser,
            io_adapter,
            local_toolset,
            state_cls,
            context=context,
        )

        self.command_manager = CommandManager(self)
        self.io_adapter.setup(self)

    def register_commands(self, cmds: list[commands.Command]):
        self.command_manager.register_commands(cmds)

    async def start(self):
        """Start the agent with optional input generator"""
        while True:
            try:
                user_input = await self.io_channel.wait_reply(None)
                if isinstance(user_input, EOFError):
                    break
                if not user_input.strip():
                    continue
                is_command = await self.command_manager.try_execute_command(user_input)
                if not is_command:
                    await self.step(user_input)
            except Exception as e:
                await self.io_channel.send(f"An error occurred: {e}")
                await self.io_channel.send("Do you want to continue? (y/n)")
                reply = await self.io_channel.wait_reply(None)
                if isinstance(reply, EOFError) or reply.strip().lower() != "y":
                    break
