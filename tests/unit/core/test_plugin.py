from types import SimpleNamespace

from arox.core.config import AgentConfig
from arox.core.plugin import Plugin, load_plugins, tool


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
    assert set(toolset.tools) == {"regular_tool", "sequential_tool"}
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
        "regular_tool",
        "sequential_tool",
    }


def test_build_toolset_returns_none_without_decorated_methods():
    plugin = Plugin(SimpleNamespace())

    assert plugin._build_toolset() is None
