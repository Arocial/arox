import asyncio
import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import fastmcp
from httpx import AsyncClient, HTTPStatusError, Timeout, TransportError
from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    FunctionToolset,
    ModelRequestContext,
    ModelSettings,
    RunContext,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
)
from pydantic_ai.models import infer_model
from pydantic_ai.providers import (
    Provider,
    gateway,
    google,
    google_cloud,
    infer_provider_class,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from tenacity import (
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arox import utils
from arox.core.config import AgentConfig, Config
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
    IOHost,
)
from arox.core.plugin import CommandManager, load_plugins
from arox.core.session import (
    USER_INPUT_ID_KEY,
    AgentSession,
)
from arox.core.skills import build_skill_catalog, discover_skills
from arox.core.slot import (
    BaseSlot,
    DiscardSlot,
    FirstSlot,
    ListSlot,
    ResultAggregator,
)
from arox.core.slot import (
    Provider as SlotProvider,
)
from arox.plugins.slots import (
    AGENT_RESET,
)

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
                fallback_strategy=wait_exponential(multiplier=2, max=30)
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
def infer_provider(
    provider: str,
    base_url: str = "",
    session_id_fn: Callable[[], str] | None = None,
    session_header: str = "",
) -> Provider[Any]:
    """Infer the provider from the provider name."""

    async def _add_session_header(request):
        session_id = session_id_fn() if session_id_fn else ""
        if session_id and session_header:
            request.headers[session_header] = session_id

    client = create_retrying_client(
        timeout=Timeout(timeout=80),
        extra_request_hooks=[_add_session_header],
    )

    kwargs: dict[str, Any] = {"http_client": client}
    if base_url:
        kwargs["base_url"] = base_url

    if provider.startswith("gateway/"):
        upstream_provider = provider.removeprefix("gateway/")
        return gateway.gateway_provider(upstream_provider, **kwargs)
    elif provider in ("google-vertex", "google-gla"):
        # Google GenAI SDK uses HttpOptions.timeout for both the httpx
        # per-request timeout AND the X-Server-Timeout header sent to the
        # server. pydantic_ai reads the httpx client's timeout and forwards
        # it to HttpOptions.timeout, so they are always coupled.
        #
        # To decouple them we:
        # 1. Set timeout to 40, which is set for both client and server timeout by genai sdk.
        # 2. Then use an httpx request event hook to remove the X-Server-Timeout
        #    header before the request is sent, so the server is not
        #    constrained by that deadline.
        async def _remove_server_timeout(request):
            request.headers.pop("X-Server-Timeout", None)

        client = create_retrying_client(
            timeout=80,
            extra_request_hooks=[_remove_server_timeout, _add_session_header],
        )
        kwargs["http_client"] = client
        if provider == "google-vertex":
            return google_cloud.GoogleCloudProvider(**kwargs)
        return google.GoogleProvider(**kwargs)
    else:
        provider_class = infer_provider_class(provider)
        return provider_class(**kwargs)


def _complete_pending_tool_calls(messages: list[ModelMessage]) -> None:
    """Append synthetic tool returns for any orphan tool calls.

    When a run is cancelled mid-step, the captured history can end with a
    ``ModelResponse`` containing ``ToolCallPart``s whose matching
    ``ToolReturnPart``s were never produced. Feeding such history back to a
    provider (e.g. Anthropic) fails because every ``tool_use`` must have a
    matching ``tool_result``. This function mutates ``messages`` in place to
    append a single ``ModelRequest`` carrying synthetic ``ToolReturnPart``s
    for every orphan tool call, keeping the history valid for the next run.
    """
    returned_ids: set[str] = set()
    pending: list[tuple[str, str]] = []  # (tool_call_id, tool_name) in order
    seen_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id not in seen_ids:
                    pending.append((part.tool_call_id, part.tool_name))
                    seen_ids.add(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    returned_ids.add(part.tool_call_id)

    orphans = [(cid, name) for cid, name in pending if cid not in returned_ids]
    if not orphans:
        return

    synthetic_parts = [
        ToolReturnPart(
            tool_name=name,
            content="Tool call cancelled before completion.",
            tool_call_id=cid,
        )
        for cid, name in orphans
    ]
    messages.append(ModelRequest(parts=synthetic_parts))


@dataclass
class UserInput:
    """A unit of user input passed to :meth:`LLMBaseAgent.step`.

    ``client_message_id`` is an opaque id assigned by a client to the message that
    produced this input; it is echoed back in :class:`ServerIdMapping` so the client
    can map its own messages to backend session-event ids.
    """

    user_input: str | None = None
    client_message_id: str | None = None


@dataclass
class ServerIdMapping:
    """Maps a UI-assigned ``message_id`` to the ``event_id`` of the recorded
    user-input session event, so the UI can resolve stable backend event ids
    (used for forking) without relying on positional ordering."""

    event_id: str | None = None
    client_id: str | None = None


@dataclass
class AgentDeps:
    agent_io: IOEndpoint


@dataclass
class AgentRunInfo:
    context_tokens: int = 0
    total_tokens: int = 0
    new_message_index: int = 0


class LLMBaseAgent(IOHost):
    def __init__(
        self,
        parsed_config: Config,
        io_adapter: AbstractIOAdapter,
        session: AgentSession,
        workspace: Path | str | None = None,
    ):
        super().__init__(io_adapter)
        self.uuid = str(uuid.uuid4())
        self._slots: dict[Any, Any] = {}
        self.parsed_config = parsed_config
        self.session = session

        self.local_toolset = FunctionToolset[AgentDeps]()
        self.mcp_client = None
        self.plugins = []
        self.message_history: list[ModelMessage] = []

        self.parse_configs()
        self._restore_agent_session(session)

        self.command_manager = CommandManager(self)

        self.run_info = AgentRunInfo()
        self.result = None
        self.builtin_hooks = Hooks[AgentDeps]()
        self.builtin_hooks.on.run(self._wrap_run)
        self.builtin_hooks.on.model_request(self._wrap_model_request)
        capabilities: list[AbstractCapability[AgentDeps]] = []
        self.plugins = load_plugins(self)
        capabilities.extend(
            [cap for plugin in self.plugins for cap in plugin.capabilities()]
        )
        capabilities.append(self.builtin_hooks)

        self.pydantic_agent = Agent[AgentDeps, DeferredToolRequests | str](
            self.model,
            capabilities=capabilities,
            toolsets=self.toolsets,
            deps_type=AgentDeps,
            output_type=(DeferredToolRequests, str),
        )

    def cancel_foreground_task(self) -> None:
        """Cancel any long-running foreground task. Subclasses can override this."""
        pass

    @property
    def name(self) -> str:
        return self.session.agent_name

    @name.setter
    def name(self, value: str):
        self.session.agent_name = value

    @property
    def workspace(self) -> Path:
        return Path(self.session.workspace) if self.session.workspace else Path.cwd()

    @workspace.setter
    def workspace(self, value: Path | str | None):
        self.session.workspace = str(Path(value).absolute()) if value else None

    @property
    def llm_context_id(self) -> str:
        return self.session.llm_context_id or ""

    @llm_context_id.setter
    def llm_context_id(self, value: str):
        self.session.llm_context_id = value

    @property
    def agent_config(self) -> AgentConfig:
        return self.session.agent_config

    @agent_config.setter
    def agent_config(self, value: AgentConfig):
        self.session.agent_config = value

    @property
    def agent_source(self) -> Literal["static", "dynamic"]:
        return self.session.agent_source

    async def handle_event(
        self, ctx: RunContext["AgentDeps"], events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.agent_io.send(event)

    def get_plugin(self, plugin_cls: type) -> Any | None:
        for plugin in self.plugins:
            if isinstance(plugin, plugin_cls):
                return plugin
        return None

    def provide_slot[P: SlotProvider, R](self, slot: BaseSlot[P, R], provider: P):
        """Register a provider for a specific slot."""
        if slot not in self._slots:
            self._slots[slot] = []
        self._slots[slot].append(provider)

    @overload
    async def invoke_slot[P: SlotProvider, R](
        self, slot: ListSlot[P, R], *args: Any, **kwargs: Any
    ) -> list[R]: ...

    @overload
    async def invoke_slot[P: SlotProvider, R](
        self, slot: FirstSlot[P, R], *args: Any, **kwargs: Any
    ) -> R | None: ...

    @overload
    async def invoke_slot[P: SlotProvider](
        self, slot: DiscardSlot[P], *args: Any, **kwargs: Any
    ) -> None: ...

    @overload
    async def invoke_slot[P: SlotProvider, R](
        self, slot: BaseSlot[P, R], *args: Any, **kwargs: Any
    ) -> R: ...

    async def invoke_slot(
        self, slot: BaseSlot[Any, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Dispatch to registered providers using the slot's aggregator strategy.

        * ``DISCARD`` – invoke every provider in registration order, discard
          return values (fire-and-forget event channel).
        * ``FIRST``  – return the result of the first registered provider.
        * ``LIST``   – return the results of all registered providers as a list.
        """
        providers = self._slots.get(slot, [])
        match slot.aggregator:
            case ResultAggregator.DISCARD:
                for handler in providers:
                    result = handler(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        await result
                return None
            case ResultAggregator.FIRST:
                if not providers:
                    return None
                result = providers[0](*args, **kwargs)
                return await result if asyncio.iscoroutine(result) else result
            case ResultAggregator.LIST:
                results = []
                for handler in providers:
                    result = handler(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    results.append(result)
                return results

    def _restore_agent_session(self, agent_session: AgentSession) -> None:
        """Apply a loaded session back onto the live agent runtime."""
        self.message_history = agent_session.rebuild_message_history()
        restored_id = agent_session.rebuild_llm_context_id()
        if restored_id:
            self.llm_context_id = restored_id
        if self.model_ref:
            self.set_model(self.model_ref)

    async def __aenter__(self):
        await super().__aenter__()

        if self.mcp_client:
            await self._stack.enter_async_context(self.mcp_client)
        self._stack.push_async_callback(self.session.save)
        for plugin in self.plugins:
            await plugin.on_start()
            self._stack.push_async_callback(plugin.on_stop)
        return self

    def add_local_tool(self, func, **kwargs):
        self.local_toolset.add_function(func, **kwargs)

    def _resolve_model(self, model_ref: str):
        from arox.core.config import ModelConfig

        model_config = self.parsed_config.model.get(model_ref)
        if not model_config:
            model_config = ModelConfig(provider_model=model_ref)
        elif not model_config.provider_model:
            model_config.provider_model = model_ref

        provider_model = model_config.provider_model
        model = infer_model(
            provider_model,
            provider_factory=lambda p: infer_provider(
                p,
                base_url=self.parsed_config.provider[p].base_url
                if p in self.parsed_config.provider
                else "",
                session_id_fn=lambda: self.llm_context_id,
                session_header=self.parsed_config.provider[p].session_header
                if p in self.parsed_config.provider
                else "",
            ),
        )
        return model, model_config, provider_model

    def set_model(self, model_ref: str):
        model, model_config, provider_model = self._resolve_model(model_ref)
        merged_model_params = utils.deep_merge(
            self.agent_model_params, dict(model_config.params)
        )

        additional_prompt = ""
        for model_prompt in self.model_aware_prompts:
            if re.search(model_prompt["pattern"], model_ref):
                additional_prompt = model_prompt["prompt"]

        self.model_ref = model_ref
        self.model_config = model_config
        self.model_params = merged_model_params
        self.provider_model = provider_model
        self.additional_prompt = additional_prompt
        self.model = model

    def parse_configs(self):
        # model configs
        self.model_ref = self.agent_config.model_ref or self.parsed_config.model_ref
        fallback = (
            self.agent_config.fallback_model_ref
            or self.parsed_config.fallback_model_ref
        )
        if isinstance(fallback, str):
            fallback = [fallback] if fallback else []
        self.fallback_model_refs: list[str] = list(fallback)
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
        # Load default metadata using configargparse
        self.raw_system_prompt = self.agent_config.system_prompt

        # skills
        skills = discover_skills(self.workspace)
        allowed_skills = self.agent_config.skills
        if allowed_skills is not None:
            if isinstance(allowed_skills, str):
                allowed_skills = [allowed_skills]
            skills = {k: v for k, v in skills.items() if k in allowed_skills}

        self.skill_catalog = ""
        if skills:
            self.skill_catalog = build_skill_catalog(skills)

        # Tools and mcp servers
        self.toolsets: list[AbstractToolset[AgentDeps]] = [self.local_toolset]
        mcp_server_configs = self.parsed_config.mcp_servers
        allowed_mcp_servers = self.agent_config.mcp_servers
        if allowed_mcp_servers is not None:
            if isinstance(allowed_mcp_servers, str):
                allowed_mcp_servers = [allowed_mcp_servers]
            mcp_server_configs = {
                k: v for k, v in mcp_server_configs.items() if k in allowed_mcp_servers
            }

        if mcp_server_configs:
            self.mcp_client = fastmcp.Client({"mcpServers": mcp_server_configs})
            mcp_toolset = MCPToolset[AgentDeps](self.mcp_client)
            self.toolsets.append(mcp_toolset)

    @property
    def system_prompt(self) -> str:
        prompt = utils.render_template(
            self.raw_system_prompt, config=self.parsed_config, agent=self
        )
        if self.skill_catalog:
            prompt += f"\n\n{self.skill_catalog}"
        return prompt

    async def _wrap_model_request(
        self,
        ctx: RunContext[AgentDeps],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        response = await handler(request_context)
        self.run_info.context_tokens = response.usage.total_tokens
        self.run_info.total_tokens += response.usage.total_tokens
        return response

    async def _wrap_run(
        self,
        ctx: RunContext[AgentDeps],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        from pydantic_ai._agent_graph import GraphAgentState

        self.run_info.new_message_index = len(ctx.messages)
        try:
            result = await handler()
        except Exception as error:
            messages = list(ctx.messages)
            _complete_pending_tool_calls(messages)

            state = GraphAgentState(
                message_history=messages,
                usage=ctx.usage,
                run_id=ctx.run_id or "",
                conversation_id=ctx.conversation_id or "",
                metadata=ctx.metadata,
            )
            result = AgentRunResult(
                output=error,
                _state=state,
                _new_message_index=self.run_info.new_message_index,
            )
        return result

    async def _run_inference(
        self,
        user_prompt: str | list[UserContent] | None,
        *,
        message_history: list[ModelMessage],
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> AgentRunResult[DeferredToolRequests | str]:
        """Run a single LLM inference with fallback model handling.

        Stateless w.r.t. the agent's own message_history / agent_session: the
        caller passes the already-composed ``user_prompt`` and message_history
        in and decides what to do with the result. If an exception occurs, it is
        captured in the returned AgentRunResult's metadata under the "exception" key.
        """
        primary_ref = self.model_ref
        try:
            for model_ref in [primary_ref, *self.fallback_model_refs]:
                if model_ref != self.model_ref:
                    self.set_model(model_ref)
                    await self.agent_io.send(
                        f"Primary model failed, falling back to {self.provider_model}"
                    )

                result = await self.pydantic_agent.run(
                    user_prompt,
                    model=self.model,
                    event_stream_handler=self.handle_event,
                    model_settings=ModelSettings(**self.model_params),
                    instructions=f"{self.system_prompt}\n{self.additional_prompt}",
                    message_history=message_history,
                    deps=AgentDeps(agent_io=self.agent_io),
                    deferred_tool_results=deferred_tool_results,
                )

                if isinstance(result.output, ModelAPIError):
                    logger.warning(
                        "Model %s failed (%s), trying next fallback",
                        self.provider_model,
                        result.output,
                    )
                else:
                    break
        finally:
            if self.model_ref != primary_ref:
                self.set_model(primary_ref)

        return result

    async def step(
        self,
        user_input: UserInput | str | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> AgentRunResult[DeferredToolRequests | str]:
        if isinstance(user_input, UserInput):
            text = user_input.user_input
            client_message_id = user_input.client_message_id
        else:
            text = user_input
            client_message_id = None

        user_prompt: list[UserContent] | None = None
        if text is not None:
            input_id = str(uuid.uuid4())
            self.session.record_user_input(text, input_id)
            if client_message_id:
                await self.agent_io.send(
                    ServerIdMapping(event_id=input_id, client_id=client_message_id)
                )
            user_prompt = [
                TextContent(content=text + "\n", metadata={USER_INPUT_ID_KEY: input_id})
            ]

        result = await self._run_inference(
            user_prompt,
            message_history=self.message_history,
            deferred_tool_results=deferred_tool_results,
        )
        self.result = result
        self.message_history = result.all_messages()

        usage = result.usage
        self.session.record_step(
            result.new_messages(),
            request_tokens=usage.input_tokens if usage else None,
            response_tokens=usage.output_tokens if usage else None,
        )
        await self.agent_io.send(AgentRunResultEvent(result))
        return result

    async def reset(self):
        self.message_history = []
        self.llm_context_id = str(uuid.uuid4())
        self.run_info = AgentRunInfo()
        await self.invoke_slot(AGENT_RESET)
        self.session.record_reset(self.llm_context_id)


class MainAgent(LLMBaseAgent, ABC):
    @abstractmethod
    async def run(self):
        pass


class DelegatableAgent(LLMBaseAgent, ABC):
    """Marker mixin for subagents that can be delegated tasks.

    Subagents inheriting from this class are exposed to the main agent as a
    callable tool and via the `/agent` slash command. The default `run_task`
    drives the agent through a single `step` and returns its textual output.
    """

    async def run_task(self, task: str) -> str | None:
        """Run a single delegated task and return its textual result."""
        result = await self.step(task)
        if result and isinstance(result.output, str):
            return result.output
        if result and isinstance(result.output, DeferredToolRequests):
            return (
                f"Sub-agent {self.name} requested deferred tools, "
                "which is not supported in delegation yet."
            )
        return None
