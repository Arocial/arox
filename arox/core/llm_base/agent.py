import asyncio
import contextlib
import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import fastmcp
from anyio import ClosedResourceError, EndOfStream
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
    UsageLimits,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    UserContent,
)
from pydantic_ai.models import infer_model

from arox import utils
from arox.core.config import AgentConfig, Config, ConfigLoader
from arox.core.io import (
    AbstractIOAdapter,
    IOEndpoint,
    IOHost,
)
from arox.core.plugin import CommandManager, load_plugins
from arox.core.session import (
    AgentRunInfo,
    AgentSession,
    SessionStatus,
)
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
from arox.core.types import AgentInfoUpdate, ServerIdMapping, UserInput
from arox.plugins.slots import (
    AGENT_RESET,
    SYSTEM_PROMPT,
)

from ._pydantic_ai_hack import infer_provider

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    agent_io: IOEndpoint
    agent: "LLMBaseAgent"


class LLMBaseAgent(IOHost):
    def __init__(
        self,
        parent_config_loader: ConfigLoader,
        io_adapter: AbstractIOAdapter,
        session: AgentSession,
    ):
        super().__init__(io_adapter)
        self.session = session
        self.uuid = session.id
        self._slots: dict[Any, Any] = {}
        self.config_loader = parent_config_loader.for_workspace(self.workspace)

        self.local_toolset = FunctionToolset[AgentDeps]()
        self.mcp_client = None
        self.plugins = []
        self.message_history: list[ModelMessage] = []
        self.message_history_fallback: list[ModelMessage] = []
        self.new_message_index = 0

        self.parse_configs()
        self._restore_agent_session(session)

        self.command_manager = CommandManager(self)

        self.result = None
        self.builtin_hooks = Hooks[AgentDeps]()
        self.builtin_hooks.on.before_run(self._before_run)
        self.builtin_hooks.on.before_model_request(self._before_model_request)
        self.builtin_hooks.on.run_error(self._on_run_error)
        self.builtin_hooks.on.model_request(self._wrap_model_request)
        capabilities: list[AbstractCapability[AgentDeps]] = []
        self.plugins = load_plugins(self)
        capabilities.extend(
            [cap for plugin in self.plugins for cap in plugin.capabilities()]
        )
        capabilities.append(self.builtin_hooks)

        self.pydantic_agent = Agent[AgentDeps, str](
            self.model,
            instructions=self.system_prompt,
            capabilities=capabilities,
            toolsets=self.toolsets,
            deps_type=AgentDeps,
            output_type=str,
        )
        self.session.initialized = True

    async def broadcast_agent_info(self):
        info = AgentInfoUpdate(agent_uuid=self.uuid)
        await self.agent_io.send(info)

    def cancel_foreground_task(self) -> None:
        """Cancel any long-running foreground task. Subclasses can override this."""
        pass

    def reload_config(self) -> Config:
        config = self.config_loader.reload()
        self.parse_configs()
        return config

    @property
    def config(self) -> Config:
        return self.config_loader.current_config

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
    def agent_config(self) -> AgentConfig:
        agent_config = self.config.agent.get(self.name)
        if not agent_config:
            raise ValueError(f"Agent config for '{self.name}' not found")
        return agent_config

    @property
    def agent_type(self) -> str:
        return self.session.agent_type

    @property
    def agent_source(self) -> Literal["static", "dynamic"]:
        return self.session.agent_source

    @property
    def status(self) -> SessionStatus:
        return self.session.status

    def close_session(self) -> None:
        self.session.close_session()

    @property
    def run_info(self) -> AgentRunInfo:
        return self.session.run_info

    @run_info.setter
    def run_info(self, value: AgentRunInfo):
        self.session.run_info = value

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
                    if result:
                        results.append(result)
                return results

    def _restore_agent_session(self, agent_session: AgentSession) -> None:
        """Apply a loaded session back onto the live agent runtime."""
        self.message_history = agent_session.rebuild_message_history()
        restored_id = agent_session.rebuild_llm_context_id()
        if restored_id:
            self.run_info.llm_context_id = restored_id
        if self.model_ref:
            self.set_model(self.model_ref)

    async def __aenter__(self):
        await super().__aenter__()
        self.session.runtime = self
        self.session.status = SessionStatus.ACTIVE
        await self.io_adapter.register_host(self)

        if self.mcp_client:
            await self._stack.enter_async_context(self.mcp_client)
        for plugin in self.plugins:
            await plugin.on_start()
            self._stack.push_async_callback(plugin.on_stop)
        await self.broadcast_agent_info()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.session.status == SessionStatus.ACTIVE:
                if exc_type is asyncio.CancelledError:
                    self.session.record_interrupted()
                elif exc_val is not None:
                    self.session.record_error(exc_val)
                else:
                    self.session.status = SessionStatus.IDLE
            with contextlib.suppress(ClosedResourceError, EndOfStream):
                await self.broadcast_agent_info()
        finally:
            try:
                await super().__aexit__(exc_type, exc_val, exc_tb)
            finally:
                if hasattr(self.io_adapter, "hosts"):
                    self.io_adapter.hosts.pop(self.uuid, None)
                self.session.runtime = None
                await self.session.save()

    def add_local_tool(self, func, **kwargs):
        self.local_toolset.add_function(func, **kwargs)

    def _resolve_model(self, model_ref: str):
        from arox.core.config import ModelConfig

        config = self.config
        model_config = config.model.get(model_ref)
        if not model_config:
            model_config = ModelConfig(provider_model=model_ref)
        else:
            model_config = model_config.model_copy(deep=True)
            if not model_config.provider_model:
                model_config.provider_model = model_ref

        provider_model = model_config.provider_model
        model = infer_model(
            provider_model,
            provider_factory=lambda p: infer_provider(
                p,
                base_url=config.provider[p].base_url if p in config.provider else "",
                run_info=self.run_info,
                session_header=config.provider[p].session_header
                if p in config.provider
                else "X-Session-Id",
                turn_header=config.provider[p].turn_header
                if p in config.provider
                else "X-Turn-Id",
            ),
        )
        return model, model_config, provider_model

    def override_model(self, model_ref: str):
        """Manually override the model. Takes precedence over config."""
        self.session.extra["model_override"] = model_ref
        self.set_model(model_ref)

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
        self.provider_name = provider_model.partition(":")[0]
        self.additional_prompt = additional_prompt
        self.model = model

    def build_skill_prompts(self, skill_names: list[str]) -> list[str]:
        """Build XML prompt blocks for the specified skills."""
        prompts = []
        for skill_name in skill_names:
            skill = self.config.skills.get(skill_name)
            if skill:
                try:
                    with open(skill["location"], "r", encoding="utf-8") as f:
                        content = f.read()
                        prompts.append(
                            f'<skill name="{skill["name"]}" location="{skill["location"]}">\n{content}\n</skill>'
                        )
                except Exception as e:
                    logger.warning(f"Failed to read skill {skill_name}: {e}")
            else:
                logger.warning(f"Skill {skill_name} not found in available skills")
        return prompts

    @staticmethod
    def _build_skill_catalog(skills: dict) -> str:
        """Build the skill catalog XML string."""
        if not skills:
            return ""

        catalog = ["<available_skills>"]
        for skill in skills.values():
            catalog.append("  <skill>")
            catalog.append(f"    <name>{skill['name']}</name>")
            catalog.append(f"    <description>{skill['description']}</description>")
            catalog.append(f"    <location>{skill['location']}</location>")
            catalog.append("  </skill>")
        catalog.append("</available_skills>")

        instructions = """
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, use your file-read tool to load
the SKILL.md at the listed location before proceeding.
When a skill references relative paths, resolve them against the skill's
directory (the parent of SKILL.md) and use absolute paths in tool calls.
"""
        return instructions + "\n" + "\n".join(catalog)

    def parse_configs(self):
        # model configs
        override = self.session.extra.get("model_override")
        self.model_ref = (
            override or self.agent_config.model_ref or self.config.model_ref
        )
        fallback = (
            self.agent_config.fallback_model_ref or self.config.fallback_model_ref
        )
        if isinstance(fallback, str):
            fallback = [fallback] if fallback else []
        self.fallback_model_refs: list[str] = list(fallback)
        self.request_limit = self.agent_config.request_limit
        self.request_limit_prompt = self.agent_config.request_limit_prompt
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
        skills = self.config.skills
        allowed_skills = self.agent_config.skills
        if allowed_skills is not None:
            if isinstance(allowed_skills, str):
                allowed_skills = [allowed_skills]
            skills = {k: v for k, v in skills.items() if k in allowed_skills}

        self.skill_catalog = ""
        if skills:
            self.skill_catalog = self._build_skill_catalog(skills)

        default_skills = self.agent_config.default_skills
        if default_skills is not None:
            if isinstance(default_skills, str):
                default_skills = [default_skills]
        else:
            default_skills = []

        self.default_skills = default_skills

        # Tools and mcp servers
        self.toolsets: list[AbstractToolset[AgentDeps]] = [self.local_toolset]
        mcp_server_configs = self.config.mcp_servers
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
    def system_prompt(self):

        async def _wrapper() -> str:
            prompt = self.raw_system_prompt
            if self.additional_prompt:
                prompt += f"\n{self.additional_prompt}"

            default_skill_prompts = self.build_skill_prompts(
                getattr(self, "default_skills", [])
            )

            if default_skill_prompts:
                prompt += "\n\n" + "\n\n".join(default_skill_prompts)

            if self.skill_catalog:
                prompt += f"\n\n{self.skill_catalog}"

            slot_prompts = await self.invoke_slot(SYSTEM_PROMPT)
            if slot_prompts:
                prompt += "\n\n" + "\n\n".join(slot_prompts)

            return utils.render_template(prompt, agent=self)

        return _wrapper

    async def _before_model_request(
        self,
        ctx: RunContext[AgentDeps],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.message_history_fallback = list(request_context.messages)
        return request_context

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

    async def _before_run(self, ctx: RunContext[AgentDeps]) -> None:
        self.new_message_index = len(ctx.messages)

    async def _on_run_error(
        self,
        ctx: RunContext[AgentDeps],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        from pydantic_ai._agent_graph import GraphAgentState

        state = GraphAgentState(
            message_history=self.message_history_fallback,
            usage=ctx.usage,
            run_id=ctx.run_id or "",
            conversation_id=ctx.conversation_id or "",
            metadata=ctx.metadata,
        )
        return AgentRunResult(
            output=error,
            _state=state,
            _new_message_index=self.new_message_index,
        )

    async def _run_inference(
        self,
        user_prompt: str | Sequence[UserContent] | None,
        *,
        message_history: list[ModelMessage],
    ) -> AgentRunResult[str]:
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

                self.run_info.run_id = str(uuid.uuid4())
                result = await self.pydantic_agent.run(
                    user_prompt,
                    model=self.model,
                    event_stream_handler=self.handle_event,
                    model_settings=ModelSettings(**self.model_params),
                    message_history=message_history,
                    usage_limits=UsageLimits(request_limit=self.request_limit),
                    deps=AgentDeps(agent_io=self.agent_io, agent=self),
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
    ) -> AgentRunResult[str]:
        if not isinstance(user_input, UserInput):
            user_input = UserInput(input_content=user_input)

        input_content = user_input.input_content

        if user_input.input_content is not None:
            self.session.record_user_input(user_input)
            if user_input.client_message_id:
                await self.agent_io.send(
                    ServerIdMapping(
                        server_message_id=user_input.server_message_id,
                        client_message_id=user_input.client_message_id,
                    )
                )

        self.reload_config()
        while True:
            result = await self._run_inference(
                input_content,
                message_history=self.message_history,
            )
            self.message_history = result.all_messages()
            self.session.record_step(result.new_messages())

            if not (
                isinstance(result.output, UsageLimitExceeded)
                and self.request_limit_prompt
            ):
                break
            logger.info("Continuing agent run after soft usage limit.")
            input_content = self.request_limit_prompt

        self.result = result
        await self.agent_io.send(AgentRunResultEvent(result))
        return result

    async def reset(self):
        self.message_history = []
        self.session.initialized = False
        self.run_info = AgentRunInfo()
        self.run_info.llm_context_id = str(uuid.uuid4())
        await self.invoke_slot(AGENT_RESET)
        self.session.record_reset(self.run_info.llm_context_id)
        self.session.initialized = True


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
        """Run a delegated task autonomously until completion."""
        result = await self.step(task)
        if result and isinstance(result.output, str):
            return result.output
        return None
