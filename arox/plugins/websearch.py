from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WebFetch, WebSearch
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    WebFetchTool,
    WebSearchTool,
)

from arox.core.plugin import Plugin

NativeToolT = TypeVar("NativeToolT", bound=AbstractNativeTool)


class WebSearchPlugin(Plugin):
    """Provide provider-native web tools with local fallbacks."""

    def _prepare_native(
        self, tool: NativeToolT
    ) -> Callable[[RunContext[Any]], NativeToolT | None]:
        def prepare(ctx: RunContext[Any]) -> NativeToolT | None:
            runtime = ctx.deps.runtime
            provider_config = runtime.config.provider.get(runtime.provider_name)
            if (
                provider_config is not None
                and tool.kind in provider_config.disabled_native_tools
            ):
                return None
            return tool

        return prepare

    def capabilities(self) -> Sequence[AbstractCapability[Any]]:
        return [
            *super().capabilities(),
            WebSearch(
                native=self._prepare_native(WebSearchTool()),
                local=True,
            ),
            WebFetch(
                native=self._prepare_native(WebFetchTool()),
                local=True,
            ),
        ]
