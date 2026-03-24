import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.completion import Completer, Completion

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    func: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)


def tool(**kwargs):
    """Decorator to register a method as a tool."""

    def decorator(func):
        func.__is_tool__ = True
        func.__tool_kwargs__ = kwargs
        return func

    return decorator


def command(name: str | list[str], description: str = ""):
    """Decorator to register a method as a command."""

    def decorator(func):
        func.__is_command__ = True
        func.__command_names__ = [name] if isinstance(name, str) else name
        func.__command_description__ = description
        return func

    return decorator


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


class PluginCommandWrapper(Command):
    def __init__(self, agent, func, names, description):
        super().__init__(agent)
        self.func = func
        self.names = names
        self.description = description

    def slashes(self) -> list[str]:
        return self.names

    async def execute(self, name: str, arg: str):
        if inspect.iscoroutinefunction(self.func):
            await self.func(name, arg)
        else:
            self.func(name, arg)

    def get_completions(self, name, args):
        # If the wrapped function has a get_completions method, call it
        # Or if the plugin has a get_completions method, call it
        if hasattr(self.func, "get_completions"):
            yield from self.func.get_completions(name, args)
        elif hasattr(self.func.__self__, "get_completions"):
            yield from self.func.__self__.get_completions(name, args)
        else:
            yield from []


class Plugin:
    def __init__(self, agent):
        self.agent = agent

    def commands(self) -> list[Command]:
        """Return a list of Command instances."""
        cmds = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "__is_command__", False):
                cmds.append(
                    PluginCommandWrapper(
                        self.agent,
                        method,
                        getattr(method, "__command_names__", []),
                        getattr(method, "__command_description__", ""),
                    )
                )
        return cmds

    def tools(self) -> list[ToolDef]:
        """Return a list of ToolDef instances."""
        tls = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "__is_tool__", False):
                tls.append(
                    ToolDef(func=method, kwargs=getattr(method, "__tool_kwargs__", {}))
                )
        return tls

    async def history_processor(self, messages: list[Any]) -> list[Any]:
        """Process message history before sending to the model."""
        return messages
