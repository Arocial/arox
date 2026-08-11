import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import update_wrapper
from types import FunctionType
from typing import Any, ClassVar

from pydantic_ai import FunctionToolset, ModelMessage, RunContext
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory, Toolset

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
class CommandSpec:
    """Binding between a :class:`CommandEvent` subclass and its handler."""

    event_cls: type[CommandEvent]
    handler: Callable[[Any], Any]
    completer: CompletionProvider | None = None


class _ToolRegistration:
    """Descriptor holding the information needed to register a plugin tool."""

    def __init__(
        self,
        func: FunctionType,
        *,
        sequential: bool,
        enabled: bool | Callable[[Any], bool],
    ) -> None:
        self.func = func
        self.sequential = sequential
        self.enabled = enabled
        update_wrapper(self, func)

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        if instance is None:
            return self
        return self.func.__get__(instance, owner)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(
    dynamic_context: Callable[[], dict[str, Any]] | None = None,
    *,
    sequential: bool = False,
    enabled: bool | Callable[[Any], bool] = True,
):
    """Prepare a plugin method for toolset registration.

    ``enabled`` can be a fixed boolean or a predicate receiving the plugin
    instance, allowing exposure to depend on configured plugin state.
    """

    def decorator(func: FunctionType) -> _ToolRegistration:
        if dynamic_context:
            from arox import utils

            context = dynamic_context()
            func.__doc__ = utils.render_template(func.__doc__, **context)
        return _ToolRegistration(func, sequential=sequential, enabled=enabled)

    return decorator


class CommandManager:
    def __init__(self, runtime):
        self.runtime = runtime
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
            self.runtime.session.record_command(name, arg)
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
    def __init__(self, runtime):
        self.runtime = runtime
        self.config: dict[str, Any] = {}

    def configure(self, config: dict[str, Any]) -> None:
        """Apply the configuration associated with this plugin instance."""
        self.config = dict(config)

    async def on_start(self) -> None:
        """Resource hook called when the runtime starts (sets up the context stack)."""

    async def on_stop(self) -> None:
        """Resource hook called when the runtime stops (torn down in reverse order)."""

    def on_load(self) -> None:
        """Wire the plugin into the runtime after construction.

        Called once after the plugin is constructed and its :meth:`commands`
        are registered (see :func:`load_plugins`). Override to register
        push-style slot providers via ``self.runtime.provide_slot(slot, handler)``
        or to perform any other one-time runtime wiring.
        """

    def commands(self) -> Sequence[CommandSpec]:
        """Return :class:`CommandSpec` bindings to register.

        Override in subclasses to wire :class:`CommandEvent` subclasses to
        plugin handler methods (and optional completion providers).
        """
        return []

    def capabilities(self) -> Sequence[AbstractCapability[Any]]:
        """Return pydantic_ai capabilities contributed by this plugin.

        The base implementation collects methods decorated with :func:`tool` into
        a ``Toolset`` capability and, when :meth:`history_processor` is overridden,
        adds a ``ProcessHistory`` capability. Subclasses needing more (tool/model/output
        wrappers, native tools, ...) should override this and extend
        ``super().capabilities()``.
        """
        caps: list[AbstractCapability[Any]] = []
        toolset = self._build_toolset()
        if toolset is not None:
            caps.append(Toolset(toolset))
        if type(self).history_processor is not Plugin.history_processor:
            caps.append(ProcessHistory(self.history_processor))
        return caps

    def _build_toolset(self) -> FunctionToolset | None:
        """Build a ``FunctionToolset`` from methods decorated with :func:`tool`."""
        toolset = FunctionToolset()
        registrations = inspect.getmembers(
            type(self), predicate=lambda member: isinstance(member, _ToolRegistration)
        )
        for _, registration in registrations:
            enabled = registration.enabled
            if not (enabled(self) if callable(enabled) else enabled):
                continue
            toolset.add_function(
                registration.__get__(self, type(self)),
                sequential=registration.sequential,
            )
        return toolset if toolset.tools else None

    async def history_processor(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Process message history before sending to the model."""
        return messages


def load_plugins(runtime) -> list[Plugin]:
    """Instantiate and wire up the plugins configured for ``runtime``.

    For each plugin class in ``runtime.agent_config.plugins``: import and
    construct it, register its slash commands, then call
    :meth:`Plugin.on_load` so it can wire up slot providers and any other
    one-time runtime hooks. Tools and capabilities are gathered separately by
    the runtime from :meth:`Plugin.capabilities`.
    """
    from arox import utils

    plugins: list[Plugin] = []
    for plugin_path in runtime.agent_config.plugins:
        plugin_cls = utils.import_class(plugin_path, group="arox.plugins")
        plugin = plugin_cls(runtime)
        plugin.configure(runtime.agent_config.plugin_config.get(plugin_path, {}))
        plugins.append(plugin)

        for spec in plugin.commands():
            runtime.command_manager.register(
                spec.event_cls, spec.handler, spec.completer
            )
        plugin.on_load()
    return plugins
