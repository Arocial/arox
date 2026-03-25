import asyncio
import contextlib
import logging
import re
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fastmcp
from httpx import AsyncClient, HTTPStatusError, TransportError
from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentRunResult,
    AgentStreamEvent,
    FunctionToolset,
    ModelSettings,
    RunContext,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import infer_model
from pydantic_ai.providers import Provider, gateway, google, infer_provider_class
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from tenacity import (
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arox import utils
from arox.agent_patterns.example_parser import parse_example_yaml
from arox.agent_patterns.hooks import PostStepHook, PreStepHook
from arox.agent_patterns.skills import build_skill_catalog, discover_skills
from arox.ui.io import AgentIOInterface

logger = logging.getLogger(__name__)


def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 499, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    async def log_request(request):
        logger.info(f"Sending request: {request.method} {request.url}")

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type(
                (HTTPStatusError, TransportError, ConnectionError, ModelHTTPError)
            ),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=60), max_wait=300
            ),
            stop=stop_after_attempt(7),
            # Re-raise the last exception if all retries fail
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ),
        validate_response=should_retry_status,
    )
    return AsyncClient(
        transport=transport,
        timeout=30.0,
        event_hooks={"request": [log_request]},
    )


# Copyied from pydantic_ai.providers.infer_provider and add http_client parameter.
def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    client = create_retrying_client()
    if provider.startswith("gateway/"):
        upstream_provider = provider.removeprefix("gateway/")
        return gateway.gateway_provider(upstream_provider)
    elif provider in ("google-vertex", "google-gla"):
        return google.GoogleProvider(
            vertexai=provider == "google-vertex", http_client=client
        )
    else:
        provider_class = infer_provider_class(provider)
        return provider_class(http_client=client)  # type: ignore


@dataclass
class AgentDeps:
    agent_io: AgentIOInterface


class LLMBaseAgent:
    def __init__(
        self,
        name,
        config_parser,
        agent_io: AgentIOInterface,
        local_toolset: FunctionToolset[AgentDeps] | None = None,
    ):
        self.uuid = str(uuid.uuid4())
        self.name = name
        self._capabilities: dict[Any, Any] = {}
        self.model_ref = None
        self.additional_prompt = ""

        self.config_parser = config_parser
        self.config = self.parse_configs()

        # Manage tools
        self.local_toolset = local_toolset
        toolsets: list[AbstractToolset[AgentDeps]] = (
            [local_toolset] if local_toolset else []
        )

        mcp_server_configs = self.config.mcp_servers
        self.mcp_client = None
        if mcp_server_configs:
            self.mcp_client = fastmcp.Client({"mcpServers": mcp_server_configs})
            mcp_toolset = FastMCPToolset[AgentDeps](self.mcp_client)
            toolsets.append(mcp_toolset)

        self.model = infer_model(self.provider_model, provider_factory=infer_provider)
        self.plugins = self.load_plugins()
        history_processors = [plugin.history_processor for plugin in self.plugins]

        self.pydantic_agent = Agent[AgentDeps, DeferredToolRequests | str](
            self.model,
            history_processors=history_processors,
            toolsets=toolsets,
            deps_type=AgentDeps,
            output_type=(DeferredToolRequests, str),
        )

        self.agent_io = agent_io

        self._stack = contextlib.AsyncExitStack()
        self.reset()

    async def handle_event(
        self, ctx: RunContext["AgentDeps"], events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.agent_io.agent_send(event)

    def load_plugins(self):
        plugin_classes = getattr(self.agent_config, "plugins", [])
        plugins = []
        for plugin_path in plugin_classes:
            plugin_cls = utils.import_class(plugin_path, group="arox.plugins")
            plugin = plugin_cls(self)
            plugins.append(plugin)

            # Register tools
            tools = plugin.tools()
            for tool_def in tools:
                if isinstance(tool_def, dict):
                    func = tool_def.pop("func")
                    self.add_local_tool(func, **tool_def)
                else:
                    self.add_local_tool(tool_def.func, **tool_def.kwargs)
        return plugins

    def provide_capability(self, capability: Any, provider: Any):
        """Register a provider for a specific capability."""
        if capability not in self._capabilities:
            self._capabilities[capability] = []
        self._capabilities[capability].append(provider)

    def get_capability(self, capability: Any) -> list[Any]:
        """
        Get the providers for a capability
        """
        return self._capabilities.get(capability, [])

    async def handle_task(self, task: str, main_agent: "LLMBaseAgent", **kwargs) -> Any:
        """Handle a task delegated from the main agent."""
        result = await self.step(task)
        if result and isinstance(result.output, str):
            return result.output
        return None

    async def __aenter__(self):
        await self._stack.enter_async_context(self.agent_io)
        if self.mcp_client:
            await self._stack.enter_async_context(self.mcp_client)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()

    def add_local_tool(self, func, **kwargs):
        if not self.local_toolset:
            self.local_toolset = FunctionToolset()
        self.local_toolset.add_function(func, **kwargs)

    def set_model(self, model_ref: str):
        self.model_ref = model_ref
        config_parser = self.config_parser
        model_group = config_parser.add_argument_group(name=f"model.'{self.model_ref}'")
        model_group.add_argument("provider_model", default=self.model_ref)
        config_parser.add_argument_group(
            name=f"model.'{self.model_ref}'.params", expose_raw=True
        )
        config = config_parser.parse_args()
        model_config = getattr(config.model, self.model_ref)

        model_params = model_config.params
        self.model_params = utils.deep_merge(self.agent_model_params, model_params)
        self.provider_model = model_config.provider_model
        for model_prompt in self.model_aware_prompts:
            if re.search(model_prompt["pattern"], self.model_ref):
                self.additional_prompt = model_prompt["prompt"]

        return config

    async def show_agent_info(self):
        await self.agent_io.agent_send(
            f"Using model {self.provider_model} for {self.name}"
        )

    def parse_configs(self):
        config_parser = self.config_parser
        name = self.name
        agent_group = config_parser.add_argument_group(
            name=f"agent.{name}", expose_raw=True
        )
        agent_group.add_argument("system_prompt", default="")
        agent_group.add_argument("model_ref", default="")
        config_parser.add_argument_group(
            name=f"agent.{name}.model_params", expose_raw=True
        )
        config_parser.add_argument_group(
            name=f"agent.{name}.model_prompt", expose_raw=True
        )
        config = config_parser.parse_args()

        self.workspace = Path.cwd()
        group_config = getattr(config.agent, name)
        self.agent_config = group_config

        # Load default metadata using configargparse
        self.system_prompt = utils.render_template(
            group_config.system_prompt, config=config
        )

        skills = discover_skills(self.workspace)
        if skills:
            catalog = build_skill_catalog(skills)
            self.system_prompt += f"\n\n{catalog}"

        self.example_messages = []
        examples_file = getattr(self.agent_config, "examples", None)
        if examples_file:
            examples_path = self.config_parser.find_config(Path(examples_file))
            if examples_path:
                with open(examples_path, "r") as f:
                    self.example_messages = parse_example_yaml(f.read())

        self.model_ref = group_config.model_ref or config.model_ref
        self.agent_model_params = group_config.model_params
        self.model_aware_prompts = []
        mp = group_config.model_prompt
        for k, v in mp.items():
            if not k.endswith("_pattern"):
                pattern = mp.get(f"{k}_pattern", "")
                self.model_aware_prompts.append(
                    {
                        "prompt": v,
                        "pattern": pattern,
                    }
                )

        return self.set_model(self.model_ref)

    async def _run_pre_step_hooks(self, input_content: str | None):
        if hasattr(self, "pre_step_hooks"):
            for hook in self.pre_step_hooks:
                await hook(self, input_content)

    async def _run_post_step_hooks(self, input_content: str | None, result: Any = None):
        if hasattr(self, "post_step_hooks"):
            for hook in self.post_step_hooks:
                await hook(self, input_content, result)

    async def step(
        self,
        input_content: str | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> AgentRunResult[DeferredToolRequests | str]:
        await self._run_pre_step_hooks(input_content)
        with capture_run_messages() as messages:
            try:
                result = await self.pydantic_agent.run(
                    input_content + "\n"
                    if isinstance(input_content, str)
                    else input_content,
                    event_stream_handler=self.handle_event,
                    model_settings=ModelSettings(**self.model_params),
                    instructions=f"{self.system_prompt}\n{self.additional_prompt}",
                    message_history=self.message_history,
                    deps=AgentDeps(agent_io=self.agent_io),
                    deferred_tool_results=deferred_tool_results,
                )
                self.message_history = result.all_messages()
                await self._run_post_step_hooks(input_content, result)
                return result
            except (asyncio.CancelledError, Exception):
                self.message_history = messages
                raise

    def reset(self):
        self.message_history = self.example_messages

    def add_pre_step_hook(self, hook: PreStepHook):
        if not hasattr(self, "pre_step_hooks"):
            self.pre_step_hooks: list[PreStepHook] = []
        self.pre_step_hooks.append(hook)

    def add_post_step_hook(self, hook: PostStepHook):
        if not hasattr(self, "post_step_hooks"):
            self.post_step_hooks: list[PostStepHook] = []
        self.post_step_hooks.append(hook)
