from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai import RunContext
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool
from pydantic_ai.usage import RunUsage

from arox.core.config import Config, ProviderConfig
from arox.plugins.websearch import WebSearchPlugin


def _context(
    *,
    provider_name: str,
    disabled_native_tools: list[str] | None = None,
) -> RunContext[Any]:
    provider = {
        provider_name: ProviderConfig(disabled_native_tools=disabled_native_tools or [])
    }
    agent = SimpleNamespace(
        provider_name=provider_name,
        config=Config(provider=provider),
    )
    deps = SimpleNamespace(agent=agent)
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
    )


def test_capabilities_enable_local_fallbacks():
    plugin = WebSearchPlugin(SimpleNamespace())

    capabilities = plugin.capabilities()
    search = cast(WebSearch[Any], capabilities[0])
    fetch = cast(WebFetch[Any], capabilities[1])

    assert isinstance(search, WebSearch)
    assert search.local is not None
    assert isinstance(fetch, WebFetch)
    assert fetch.local is not None


def test_native_tools_are_enabled_by_default():
    plugin = WebSearchPlugin(SimpleNamespace())
    capabilities = plugin.capabilities()
    search = cast(WebSearch[Any], capabilities[0])
    fetch = cast(WebFetch[Any], capabilities[1])
    ctx = _context(provider_name="openai")

    prepare_search = cast(Any, search.native)
    prepare_fetch = cast(Any, fetch.native)

    assert isinstance(prepare_search(ctx), WebSearchTool)
    assert isinstance(prepare_fetch(ctx), WebFetchTool)


def test_native_tools_can_be_disabled_independently():
    plugin = WebSearchPlugin(SimpleNamespace())
    capabilities = plugin.capabilities()
    search = cast(WebSearch[Any], capabilities[0])
    fetch = cast(WebFetch[Any], capabilities[1])
    ctx = _context(
        provider_name="google",
        disabled_native_tools=["web_search"],
    )

    prepare_search = cast(Any, search.native)
    prepare_fetch = cast(Any, fetch.native)

    assert prepare_search(ctx) is None
    assert isinstance(prepare_fetch(ctx), WebFetchTool)
