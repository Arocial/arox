import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai import ModelRetry

from arox.core.config import AgentConfig
from arox.core.plugin import (
    CommandEvent,
    CommandManager,
    Plugin,
    _wrap_tool_errors,
    load_plugins,
    tool,
)


class _ConfiguredPlugin(Plugin):
    pass


class _ToolPlugin(Plugin):
    @tool()
    def regular_tool(self, value: str = "ok") -> str:
        return value

    @tool(sequential=True)
    def sequential_tool(self) -> None:
        pass

    @tool(enabled=lambda plugin: plugin.config.get("conditional", False))
    def conditional_tool(self) -> None:
        pass

    @tool()
    def failing_tool(self) -> None:
        raise RuntimeError("tool failed")

    @tool(on_error="retry")
    async def retrying_tool(self) -> None:
        raise RuntimeError("try again")

    @tool(on_error="raise")
    def raising_tool(self) -> None:
        raise RuntimeError("fatal tool failure")

    @tool()
    def model_retry_tool(self) -> None:
        raise ModelRetry("fix the arguments")


class _InfoCommand(CommandEvent):
    slashes = ("info",)


class _InvalidSlashCommand(CommandEvent):
    slashes = ("invalid",)

    @classmethod
    def from_slash(cls, name: str, arg: str | None) -> None:
        return None


def test_load_plugins_applies_named_plugin_config(monkeypatch):
    monkeypatch.setattr(
        "arox.utils.import_class", lambda *_args, **_kwargs: _ConfiguredPlugin
    )
    agent = SimpleNamespace(
        agent_config=AgentConfig(
            plugins=["configured"],
            plugin_config={"configured": {"enabled": True}},
        ),
        command_manager=SimpleNamespace(register=lambda *_args: None),
    )

    plugins = load_plugins(agent)

    assert len(plugins) == 1
    assert plugins[0].config == {"enabled": True}


def test_build_toolset_registers_decorated_methods():
    plugin = _ToolPlugin(SimpleNamespace())

    toolset = plugin._build_toolset()

    assert toolset is not None
    assert plugin.regular_tool() == "ok"
    assert set(toolset.tools) == {
        "failing_tool",
        "model_retry_tool",
        "raising_tool",
        "regular_tool",
        "retrying_tool",
        "sequential_tool",
    }
    assert set(
        toolset.tools["regular_tool"].function_schema.json_schema["properties"]
    ) == {"value"}
    assert toolset.tools["regular_tool"].sequential is False
    assert toolset.tools["sequential_tool"].sequential is True

    plugin.configure({"conditional": True})
    configured_toolset = plugin._build_toolset()
    assert configured_toolset is not None
    assert set(configured_toolset.tools) == {
        "conditional_tool",
        "failing_tool",
        "model_retry_tool",
        "raising_tool",
        "regular_tool",
        "retrying_tool",
        "sequential_tool",
    }


def test_build_toolset_returns_none_without_decorated_methods():
    plugin = Plugin(SimpleNamespace())

    assert plugin._build_toolset() is None


def test_tool_returns_structured_error_by_default(caplog):
    wrapped = _wrap_tool_errors(_ToolPlugin(SimpleNamespace()).failing_tool, "return")

    result = wrapped()

    assert result == {
        "ok": False,
        "error": {
            "type": "RuntimeError",
            "message": "tool failed",
            "retryable": False,
        },
    }
    assert "Plugin tool failing_tool failed" in caplog.text


@pytest.mark.asyncio
async def test_tool_can_convert_error_to_model_retry():
    tool_func = _ToolPlugin(SimpleNamespace()).retrying_tool
    wrapped = _wrap_tool_errors(tool_func, "retry")

    assert inspect.iscoroutinefunction(wrapped)
    with pytest.raises(ModelRetry, match="try again"):
        await wrapped()


def test_tool_can_raise_error():
    wrapped = _wrap_tool_errors(_ToolPlugin(SimpleNamespace()).raising_tool, "raise")

    with pytest.raises(RuntimeError, match="fatal tool failure"):
        wrapped()


def test_tool_control_exceptions_are_not_handled():
    wrapped = _wrap_tool_errors(
        _ToolPlugin(SimpleNamespace()).model_retry_tool, "return"
    )

    with pytest.raises(ModelRetry, match="fix the arguments"):
        wrapped()


def test_tool_rejects_invalid_error_behavior():
    with pytest.raises(ValueError, match="Unsupported tool error behavior"):
        tool(on_error="invalid")  # ty: ignore[invalid-argument-type]


def _command_manager() -> CommandManager:
    runtime = SimpleNamespace(session=SimpleNamespace(record_command=Mock()))
    manager = CommandManager(runtime)
    manager.register(_InfoCommand, lambda event: "info")
    manager.register(_InvalidSlashCommand, lambda event: "unused")
    return manager


@pytest.mark.asyncio
async def test_dispatch_reports_explicit_slash_outcomes():
    manager = _command_manager()

    handled = await manager.dispatch("/info")

    assert handled.status == "handled"
    assert handled.reply is not None
    assert handled.reply.output == "info"
    assert (await manager.dispatch("hello")).status == "not_command"
    assert (await manager.dispatch("/missing")).status == "unknown"
    assert (await manager.dispatch("/invalid")).status == "invalid"


@pytest.mark.asyncio
async def test_dispatch_reports_explicit_serialized_outcomes():
    manager = _command_manager()

    handled = await manager.dispatch({"type": "_InfoCommand"})

    assert handled.status == "handled"
    assert handled.reply is not None
    assert handled.reply.output == "info"
    assert (await manager.dispatch({})).status == "invalid"
    assert (await manager.dispatch({"type": "MissingCommand"})).status == "unknown"
