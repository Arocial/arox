from types import SimpleNamespace

from arox.core.plugin import Plugin, tool


class _ToolPlugin(Plugin):
    @tool()
    def regular_tool(self, value: str = "ok") -> str:
        return value

    @tool(sequential=True)
    def sequential_tool(self) -> None:
        pass


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


def test_build_toolset_returns_none_without_decorated_methods():
    plugin = Plugin(SimpleNamespace())

    assert plugin._build_toolset() is None
