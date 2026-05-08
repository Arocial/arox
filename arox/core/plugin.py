import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic_ai import ModelMessage, RunContext

from arox.core.completion import (
    CompletionProvider,
    CompletionRouter,
)
from arox.core.io import ReplyEvent, RequestEvent

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CommandEvent(RequestEvent):
    """Base class for slash / control command events.

    Subclasses declare the slash names they answer to via :attr:`slashes`
    and (optionally) override :meth:`from_slash` to parse ``(name, arg)``
    into a populated event. The default :meth:`from_slash` calls ``cls()``
    and works for events with no required fields.
    """

    slashes: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""

    @classmethod
    def from_slash(cls, name: str, arg: str | None) -> "CommandEvent | None":
        return cls()


@dataclass(kw_only=True)
class CommandReply(ReplyEvent):
    """Reply produced by :meth:`CommandManager.execute`."""

    output: str | None = None


@dataclass
class ToolDef:
    func: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandSpec:
    """Binding between a :class:`CommandEvent` subclass and its handler."""

    event_cls: type[CommandEvent]
    handler: Callable[[Any], Any]
    completer: CompletionProvider | None = None


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


class CommandManager:
    def __init__(self, agent):
        self.agent = agent
        self.completion_router = CompletionRouter()
        self._handlers: dict[type[CommandEvent], Callable[[Any], Any]] = {}
        self._slash_map: dict[str, type[CommandEvent]] = {}

    def register(
        self,
        event_cls: type[CommandEvent],
        handler: Callable[[Any], Any],
        completer: CompletionProvider | None = None,
    ) -> None:
        """Register ``event_cls`` together with its handler (and optional completer).

        Wires up slash parsing (via ``event_cls.slashes`` +
        ``event_cls.from_slash``), structured deserialization (via
        ``event_cls.__name__``), and execution dispatch in one call.

        ``completer``, when given, is a :class:`CompletionProvider` invoked
        for argument completion of every slash declared on ``event_cls``.
        """
        self._handlers[event_cls] = handler
        for slash in event_cls.slashes:
            if slash in self._slash_map:
                logger.warning(
                    "Slash /%s already registered to %s; overwriting with %s",
                    slash,
                    self._slash_map[slash].__name__,
                    event_cls.__name__,
                )
            self._slash_map[slash] = event_cls
            self.completion_router.register_slash(
                slash, description=event_cls.description, sub=completer
            )

    @property
    def command_map(self) -> dict[str, type[CommandEvent]]:
        """Public view of slash → event class, used by adapters for suggestions."""
        return self._slash_map

    async def parse_slash_command(self, line: str) -> CommandEvent | None:
        """Parse a raw ``/<name> [arg]`` line into a :class:`CommandEvent`."""
        if not line.startswith("/"):
            return None
        parts = line[1:].split(" ", 1)
        name = parts[0]
        arg = parts[1] if len(parts) > 1 else None

        event_cls = self._slash_map.get(name)
        if event_cls is None:
            logger.warning("Command not found: /%s", name)
            return None
        try:
            self.agent.agent_session.add_event("command", {"command": name, "arg": arg})
            event = event_cls.from_slash(name, arg)
            if event is not None and not isinstance(event, CommandEvent):
                logger.warning(
                    "%s.from_slash returned non-CommandEvent value %r; ignoring",
                    event_cls.__name__,
                    type(event).__name__,
                )
                return None
            return event
        except Exception:
            logger.exception("Error parsing command /%s", name)
            return None

    async def try_handle_slash(self, line: str) -> CommandReply | None:
        """Parse and execute ``line`` if it's a slash command.

        Returns the :class:`CommandReply` on success, or ``None`` if ``line``
        is not a slash command or could not be parsed (e.g. unknown command).
        Shared entry point for IO adapters that need to intercept slash
        commands typed into a normal input field.
        """
        if not line.startswith("/"):
            return None
        event = await self.parse_slash_command(line)
        if event is None:
            return None
        return await self.execute(event)

    def deserialize_event(self, payload: dict[str, Any]) -> CommandEvent | None:
        """Reconstruct a :class:`CommandEvent` from a ``{"type", ...}`` dict."""
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


class Plugin:
    def __init__(self, agent):
        self.agent = agent

    def commands(self) -> Sequence[CommandSpec]:
        """Return :class:`CommandSpec` bindings to register.

        Override in subclasses to wire :class:`CommandEvent` subclasses to
        plugin handler methods (and optional completion providers).
        """
        return []

    def tools(self) -> list[ToolDef]:
        tls = []
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
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
