import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import update_wrapper, wraps
from types import FunctionType
from typing import Any, ClassVar, Literal, TypedDict

from pydantic_ai import (
    ApprovalRequired,
    CallDeferred,
    FunctionToolset,
    ModelMessage,
    ModelRetry,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory, Toolset
from pydantic_ai.exceptions import SkipToolExecution, SkipToolValidation

from arox.core.completion import (
    CompletionProvider,
    CompletionRouter,
)

logger = logging.getLogger(__name__)

ToolErrorBehavior = Literal["return", "retry", "raise"]
_TOOL_ERROR_MESSAGE_LIMIT = 2000
_TOOL_CONTROL_EXCEPTIONS = (
    ModelRetry,
    ApprovalRequired,
    CallDeferred,
    SkipToolExecution,
    SkipToolValidation,
)


class ToolErrorDetails(TypedDict):
    type: str
    message: str
    retryable: bool


class ToolErrorResult(TypedDict):
    ok: Literal[False]
    error: ToolErrorDetails


@dataclass(kw_only=True)
class CommandEvent:
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
class CommandReply:
    """Reply produced by :meth:`CommandManager.execute`."""

    output: str | None = None


CommandDispatchStatus = Literal["handled", "not_command", "unknown", "invalid"]


@dataclass(frozen=True)
class CommandDispatchResult:
    """Result of recognizing and dispatching an external command representation."""

    status: CommandDispatchStatus
    reply: CommandReply | None = None


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
        on_error: ToolErrorBehavior,
    ) -> None:
        self.func = func
        self.sequential = sequential
        self.enabled = enabled
        self.on_error = on_error
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
    on_error: ToolErrorBehavior = "return",
):
    """Prepare a plugin method for toolset registration.

    ``enabled`` can be a fixed boolean or a predicate receiving the plugin
    instance, allowing exposure to depend on configured plugin state. ``on_error``
    controls whether ordinary tool exceptions are returned to the model, converted
    to a model retry, or raised to terminate the current turn.
    """

    if on_error not in ("return", "retry", "raise"):
        raise ValueError(f"Unsupported tool error behavior: {on_error!r}")

    def decorator(func: FunctionType) -> _ToolRegistration:
        if dynamic_context:
            from arox import utils

            context = dynamic_context()
            func.__doc__ = utils.render_template(func.__doc__, **context)
        return _ToolRegistration(
            func,
            sequential=sequential,
            enabled=enabled,
            on_error=on_error,
        )

    return decorator


def _tool_error_message(error: Exception) -> str:
    message = str(error) or type(error).__name__
    if len(message) > _TOOL_ERROR_MESSAGE_LIMIT:
        return message[:_TOOL_ERROR_MESSAGE_LIMIT] + "..."
    return message


def _handle_tool_error(
    tool_name: str,
    behavior: ToolErrorBehavior,
    error: Exception,
) -> ToolErrorResult:
    logger.exception("Plugin tool %s failed", tool_name)
    if behavior == "raise":
        raise error

    message = _tool_error_message(error)
    if behavior == "retry":
        raise ModelRetry(message) from error

    return {
        "ok": False,
        "error": {
            "type": type(error).__name__,
            "message": message,
            "retryable": False,
        },
    }


def _wrap_tool_errors(
    func: Callable[..., Any],
    behavior: ToolErrorBehavior,
) -> Callable[..., Any]:
    tool_name = getattr(func, "__name__", type(func).__name__)
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except _TOOL_CONTROL_EXCEPTIONS:
                raise
            except Exception as error:
                return _handle_tool_error(tool_name, behavior, error)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except _TOOL_CONTROL_EXCEPTIONS:
            raise
        except Exception as error:
            return _handle_tool_error(tool_name, behavior, error)

    return sync_wrapper


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

    async def dispatch(self, command: str | dict[str, Any]) -> CommandDispatchResult:
        """Recognize and execute a textual or structured command."""
        if isinstance(command, str):
            status, event = await self._resolve_slash(command)
        else:
            status, event = self._resolve_serialized(command)

        if event is None:
            return CommandDispatchResult(status)
        return CommandDispatchResult("handled", await self.execute(event))

    async def _resolve_slash(
        self, line: str
    ) -> tuple[CommandDispatchStatus, CommandEvent | None]:
        if not line.startswith("/"):
            return "not_command", None

        name = line[1:].split(" ", 1)[0]
        if name not in self._slash_map:
            logger.warning("Command not found: /%s", name)
            return "unknown", None

        event = await self.parse_slash_command(line)
        if event is None:
            return "invalid", None
        return "handled", event

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

    def _resolve_serialized(
        self, payload: dict[str, Any]
    ) -> tuple[CommandDispatchStatus, CommandEvent | None]:
        type_name = payload.get("type")
        if not type_name:
            return "invalid", None
        if not any(evt_cls.__name__ == type_name for evt_cls in self._handlers):
            logger.warning("Unknown CommandEvent type %s", type_name)
            return "unknown", None

        event = self.deserialize_event(payload)
        if event is None:
            return "invalid", None
        return "handled", event

    async def execute(self, event: CommandEvent) -> CommandReply:
        """Run the handler for ``event`` and return a :class:`CommandReply`."""
        handler = self._handlers.get(type(event))
        if handler is None:
            logger.warning(
                "No handler registered for CommandEvent %s",
                type(event).__name__,
            )
            return CommandReply()
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, CommandReply):
                return result
            if isinstance(result, str):
                return CommandReply(output=result)
            return CommandReply()
        except Exception as e:
            logger.exception("Error executing CommandEvent %s", type(event).__name__)
            return CommandReply(output=f"Error: {e}")


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
            func = registration.__get__(self, type(self))
            toolset.add_function(
                _wrap_tool_errors(func, registration.on_error),
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
