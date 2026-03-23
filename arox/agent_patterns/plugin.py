import inspect
import logging
from typing import Any

from prompt_toolkit.completion import Completer, Completion

logger = logging.getLogger(__name__)


def parse_cmdline(cmdline):
    if not cmdline.startswith("/"):
        return None, None

    cmd = cmdline.split(" ", 1)
    c_name = cmd[0][1:]
    c_arg = cmd[1] if len(cmd) > 1 else None
    return c_name, c_arg


class CommandCompleter(Completer):
    """Main completer that delegates to specific command completers"""

    def __init__(self, manager):
        self.command_manager = manager

    def get_completions(self, document, complete_event):
        yield from self._get_completions(document.text)

    def _get_completions(self, text):
        name, args = parse_cmdline(text)
        if not name:
            return
        if args is None:  # Complete command names
            candidates = self.command_manager.command_names()
            for candidate in candidates:
                if name in candidate:
                    yield Completion(
                        candidate, start_position=-len(name), display=candidate
                    )
            return

        yield from self.command_manager.get_completions(name, args)


class Command:
    """Base class for agent commands"""

    command: str = ""
    description: str = ""

    def __init__(self, agent):
        self.agent = agent

    def slashes(self) -> list[str]:
        return [self.command]

    async def execute(self, name: str, arg: str):
        """Execute command with given input"""
        raise NotImplementedError

    def get_completions(self, name, args):
        yield from []


class CommandManager:
    def __init__(self, agent):
        self.command_map = {}
        self.agent = agent
        self.completer = CommandCompleter(self)

    def register_commands(self, commands: list[Command]):
        for command in commands:
            for s in command.slashes():
                self.command_map[s] = command

    async def try_execute_command(self, user_input: str) -> bool:
        c_name, c_arg = parse_cmdline(user_input)
        if not c_name:
            return False

        command = self.command_map.get(c_name)
        if command:
            try:
                if inspect.iscoroutinefunction(command.execute):
                    await command.execute(c_name, c_arg)
                else:
                    command.execute(c_name, c_arg)
            except Exception as e:
                await self.agent.agent_io.agent_send(
                    f"Error executing command {c_name}: {e}"
                )
        else:
            await self.agent.agent_io.agent_send(f"Command not found: {user_input}")
        return True

    def get_completions(self, name: str, args: str):
        command = self.command_map.get(name)
        if not command:
            return
        yield from command.get_completions(name, args)

    def command_names(self):
        return self.command_map.keys()


class Plugin:
    def __init__(self, agent):
        self.agent = agent

    def commands(self) -> list:
        """Return a list of Command instances."""
        return []

    def tools(self) -> list[dict[str, Any]]:
        """Return a list of dicts containing 'func' and other kwargs for add_local_tool."""
        return []
