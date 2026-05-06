import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.completion import Completer, Completion
from pydantic_ai import ModelMessage, RunContext

from arox.core.io import ReplyEvent, RequestEvent

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CommandEvent(RequestEvent):
    """Base class for slash / control command events.

    Adapters turn user-initiated commands (slash strings, structured
    payloads) into concrete subclasses via :meth:`CommandManager.parse_slash_command`
    or :meth:`CommandManager.deserialize_event`, then run them through
    :meth:`CommandManager.execute` to obtain a :class:`CommandReply` which
    the adapter renders.
    """


@dataclass(kw_only=True)
class CommandReply(ReplyEvent):
    """Reply produced by :meth:`CommandManager.execute`."""

    output: str | None = None


@dataclass
class ToolDef:
    func: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)


def tool(dynamic_context: Callable[[], dict[str, Any]] | None = None, **kwargs):
    """Decorator to register a method as a tool."""

    def decorator(func):
        func.__is_tool__ = True
        func.__tool_kwargs__ = kwargs
        if dynamic_context:
            from arox import utils

            context = dynamic_context()
            func.__doc__ = utils.render_template(func.__doc__, **context)
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


class CommandCompleter(Completer):
    """Main completer that delegates to specific command completers"""

    def __init__(self, manager):
        self.command_manager = manager

    def get_completions(self, document, complete_event):
        yield from self._get_completions(document.text)

    def _get_completions(self, text):
        if not text.startswith("/"):
            return
        parts = text[1:].split(" ", 1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else None
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

    async def to_event(self, name: str, arg: str | None) -> CommandEvent | None:
        """Turn ``(name, arg)`` into a concrete :class:`CommandEvent`.

        Return ``None`` if the command produced no event (e.g. completed
        synchronously, or had nothing to dispatch).
        """
        raise NotImplementedError

    def get_completions(self, name, args):
        yield from []


class CommandManager:
    def __init__(self, agent):
        self.command_map: dict[str, Command] = {}
        self.agent = agent
        self.completer = CommandCompleter(self)
        self._handlers: dict[type[CommandEvent], Callable[[Any], Any]] = {}

    def register_commands(self, commands: list[Command]):
        for command in commands:
            for s in command.slashes():
                self.command_map[s] = command

    def register_handler(
        self,
        event_type: type[CommandEvent],
        handler: Callable[[Any], Any],
    ) -> None:
        """Register a handler for a :class:`CommandEvent` subclass.

        Handlers may be sync or async. They should return a string (rendered
        as :attr:`CommandReply.output`), a :class:`CommandReply`, or ``None``.
        """
        self._handlers[event_type] = handler

    async def parse_slash_command(self, line: str) -> CommandEvent | None:
        """Parse a raw ``/<name> [arg]`` line into a :class:`CommandEvent`.

        Returns ``None`` if ``line`` is not a slash command, or if the named
        command is unknown / produced no event.
        """
        if not line.startswith("/"):
            return None
        parts = line[1:].split(" ", 1)
        name = parts[0]
        arg = parts[1] if len(parts) > 1 else None

        command = self.command_map.get(name)
        if command is None:
            logger.warning("Command not found: /%s", name)
            return None
        try:
            self.agent.agent_session.add_event("command", {"command": name, "arg": arg})
            result = command.to_event(name, arg)
            if inspect.iscoroutine(result):
                result = await result
            if result is not None and not isinstance(result, CommandEvent):
                logger.warning(
                    "Command %s returned non-CommandEvent value %r; ignoring",
                    name,
                    type(result).__name__,
                )
                return None
            return result
        except Exception:
            logger.exception("Error parsing command /%s", name)
            return None

    def deserialize_event(self, payload: dict[str, Any]) -> CommandEvent | None:
        """Reconstruct a :class:`CommandEvent` from a structured ``{"type", ...}`` dict.

        Looks up the event class by ``payload["type"]`` against the names of
        registered handler event types, and constructs it from the remaining
        fields. Returns ``None`` if the type is missing or unknown.
        """
        type_name = payload.get("type")
        if not type_name:
            return None
        for evt_cls in self._handlers:
            if evt_cls.__name__ == type_name:
                data = {k: v for k, v in payload.items() if k != "type"}
                try:
                    return evt_cls(**data)
                except TypeError:
                    logger.exception(
                        "Failed to construct %s from payload %r", type_name, payload
                    )
                    return None
        logger.warning("Unknown CommandEvent type %s", type_name)
        return None

    async def execute(self, event: CommandEvent) -> CommandReply:
        """Run the handler for ``event`` and return a :class:`CommandReply`."""
        handler = self._handlers.get(type(event))
        if handler is None:
            logger.warning(
                "No handler registered for CommandEvent %s",
                type(event).__name__,
            )
            return CommandReply(req_id=event.req_id)
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, CommandReply):
                result.req_id = event.req_id
                return result
            if isinstance(result, str):
                return CommandReply(req_id=event.req_id, output=result)
            return CommandReply(req_id=event.req_id)
        except Exception as e:
            logger.exception("Error executing CommandEvent %s", type(event).__name__)
            return CommandReply(req_id=event.req_id, output=f"Error: {e}")

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

    async def to_event(self, name: str, arg: str | None) -> CommandEvent | None:
        if inspect.iscoroutinefunction(self.func):
            return await self.func(name, arg)
        return self.func(name, arg)

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

    async def history_processor(
        self,
        ctx: RunContext[None],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Process message history before sending to the model."""
        return messages

    async def get_info(self) -> str:
        """Return information to be displayed by the /info command."""
        return ""
