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
from httpx import AsyncClient, HTTPStatusError, Timeout, TransportError
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
from arox.core.config import AgentConfig, Config
from arox.core.hooks import PostStepHook, PreStepHook
from arox.core.session import AgentSession, _deserialize_messages, _serialize_messages
from arox.core.skills import build_skill_catalog, discover_skills
from arox.ui.io import AgentIOInterface

logger = logging.getLogger(__name__)


def create_retrying_client(extra_request_hooks=None, **client_args):
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
                (HTTPStatusError, TransportError, ConnectionError)
            ),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=60), max_wait=300
            ),
            stop=stop_after_attempt(8),
            # Re-raise the last exception if all retries fail
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ),
        validate_response=should_retry_status,
    )
    request_hooks = [log_request] + (extra_request_hooks or [])
    return AsyncClient(
        transport=transport,
        event_hooks={"request": request_hooks},
        **client_args,
    )


# Copyied from pydantic_ai.providers.infer_provider and add http_client parameter.
def infer_provider(provider: str, base_url: str = "") -> Provider[Any]:
    """Infer the provider from the provider name."""
    client = create_retrying_client(
        timeout=Timeout(timeout=20),
    )
    if provider.startswith("gateway/"):
        upstream_provider = provider.removeprefix("gateway/")
        return gateway.gateway_provider(upstream_provider)
    elif provider in ("google-vertex", "google-gla"):
        # Google GenAI SDK uses HttpOptions.timeout for both the httpx
        # per-request timeout AND the X-Server-Timeout header sent to the
        # server. pydantic_ai reads the httpx client's timeout and forwards
        # it to HttpOptions.timeout, so they are always coupled.
        #
        # To decouple them we:
        # 1. Set timeout to 20, which is set for both client and server timeout by genai sdk.
        # 2. Then use an httpx request event hook to remove the X-Server-Timeout
        #    header before the request is sent, so the server is not
        #    constrained by that deadline.
        async def _remove_server_timeout(request):
            request.headers.pop("X-Server-Timeout", None)

        client = create_retrying_client(
            timeout=20,
            extra_request_hooks=[_remove_server_timeout],
        )
        return google.GoogleProvider(
            vertexai=provider == "google-vertex", http_client=client
        )
    else:
        provider_class = infer_provider_class(provider)
        kwargs: dict[str, Any] = {"http_client": client}
        if base_url:
            kwargs["base_url"] = base_url
        return provider_class(**kwargs)  # type: ignore


@dataclass
class AgentDeps:
    agent_io: AgentIOInterface


class LLMBaseAgent:
    def __init__(
        self,
        name: str,
        parsed_config: Config,
        agent_io: AgentIOInterface,
        local_toolset: FunctionToolset[AgentDeps] | None = None,
        workspace: Path | str | None = None,
    ):
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.workspace = Path(workspace).absolute() if workspace else Path.cwd()
        self.agent_session: AgentSession = AgentSession(agent_name=name)
        self._capabilities: dict[Any, Any] = {}
        self.model_ref = None
        self.additional_prompt = ""

        self.parsed_config = parsed_config
        self.agent_config: AgentConfig = parsed_config.agent.get(name) or AgentConfig()

        # Manage tools
        self.local_toolset = local_toolset
        toolsets: list[AbstractToolset[AgentDeps]] = (
            [local_toolset] if local_toolset else []
        )

        mcp_server_configs = self.parsed_config.mcp_servers
        self.mcp_client = None
        if mcp_server_configs:
            self.mcp_client = fastmcp.Client({"mcpServers": mcp_server_configs})
            mcp_toolset = FastMCPToolset[AgentDeps](self.mcp_client)
            toolsets.append(mcp_toolset)

        self.parse_configs()

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
        plugin_classes = self.agent_config.plugins
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
        main_agent.agent_session.add_event(
            "subagent_call",
            {"subagent": self.name, "task": task},
        )
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
        model_config = self.parsed_config.model.get(model_ref)
        if not model_config:
            from arox.core.config import ModelConfig

            model_config = ModelConfig(provider_model=model_ref)

        model_params = model_config.params
        merged_model_params = utils.deep_merge(self.agent_model_params, model_params)
        provider_model = model_config.provider_model
        base_url = model_config.base_url

        additional_prompt = ""
        for model_prompt in self.model_aware_prompts:
            if re.search(model_prompt["pattern"], model_ref):
                additional_prompt = model_prompt["prompt"]

        model = infer_model(
            provider_model,
            provider_factory=lambda p: infer_provider(p, base_url=base_url),
        )

        self.model_ref = model_ref
        self.model_params = merged_model_params
        self.provider_model = provider_model
        self.base_url = base_url
        self.additional_prompt = additional_prompt
        self.model = model

    async def show_agent_info(self):
        await self.agent_io.agent_send(
            f"Using model {self.provider_model} for {self.name}"
        )

    def parse_configs(self):
        # Load default metadata using configargparse
        self.system_prompt = utils.render_template(
            self.agent_config.system_prompt, config=self.parsed_config, agent=self
        )

        skills = discover_skills(self.workspace)
        allowed_skills = self.agent_config.skills
        if allowed_skills is not None:
            if isinstance(allowed_skills, str):
                allowed_skills = [allowed_skills]
            skills = {k: v for k, v in skills.items() if k in allowed_skills}

        if skills:
            catalog = build_skill_catalog(skills)
            self.system_prompt += f"\n\n{catalog}"

        self.example_messages = []
        examples_data = self.agent_config.examples
        if examples_data:
            self.example_messages = _deserialize_messages(examples_data)

        self.model_ref = self.agent_config.model_ref or self.parsed_config.model_ref
        self.agent_model_params = self.agent_config.model_params
        self.model_aware_prompts = []
        mp = self.agent_config.model_prompt
        for k, v in mp.items():
            if not k.endswith("_pattern"):
                pattern = mp.get(f"{k}_pattern", "")
                self.model_aware_prompts.append(
                    {
                        "prompt": v,
                        "pattern": pattern,
                    }
                )

        self.set_model(self.model_ref)

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
                    model=self.model,
                    event_stream_handler=self.handle_event,
                    model_settings=ModelSettings(**self.model_params),
                    instructions=f"{self.system_prompt}\n{self.additional_prompt}",
                    message_history=self.message_history,
                    deps=AgentDeps(agent_io=self.agent_io),
                    deferred_tool_results=deferred_tool_results,
                )
                self.message_history = result.all_messages()
                self._record_step_event(input_content, result)
                await self._run_post_step_hooks(input_content, result)
                return result
            except (asyncio.CancelledError, Exception):
                self.message_history = messages
                raise

    def _record_step_event(
        self,
        input_content: str | None,
        result: AgentRunResult[Any],
    ):
        new_messages = result.new_messages()
        usage = result.usage()
        self.agent_session.add_event(
            "agent_step",
            {
                "input": input_content,
                "new_messages": _serialize_messages(new_messages),
                "request_tokens": usage.input_tokens if usage else None,
                "response_tokens": usage.output_tokens if usage else None,
            },
        )

    def restore_session(self, agent_session: AgentSession):
        self.agent_session = agent_session
        self.message_history = agent_session.rebuild_message_history(
            self.example_messages
        )

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
