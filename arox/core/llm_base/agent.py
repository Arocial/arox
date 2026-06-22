import asyncio
import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import fastmcp
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
)
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextContent,
    UserContent,
)
from pydantic_ai.models import infer_model
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

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
    AgentRunInfo,
    AgentSession,
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
from arox.plugins.slots import (
    AGENT_RESET,
    SYSTEM_PROMPT,
)

from ._pydantic_ai_hack import infer_provider
from .types import ServerIdMapping, UserInput

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    agent_io: IOEndpoint
    agent: "LLMBaseAgent"


class LLMBaseAgent(IOHost):
    def __init__(
        self,
        parsed_config: Config,
        io_adapter: AbstractIOAdapter,
        session: AgentSession,
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
        self.message_history_fallback: list[ModelMessage] = []
        self.new_message_index = 0

        self.parse_configs()
        self._restore_agent_session(session)

        self.command_manager = CommandManager(self)

        self.result = None
        self.builtin_hooks = Hooks[AgentDeps]()
        self.builtin_hooks.on.before_run(self._before_run)
        self.builtin_hooks.on.run_error(self._on_run_error)
        self.builtin_hooks.on.model_request(self._wrap_model_request)
        capabilities: list[AbstractCapability[AgentDeps]] = []
        self.plugins = load_plugins(self)
        capabilities.extend(
            [cap for plugin in self.plugins for cap in plugin.capabilities()]
        )
        capabilities.append(self.builtin_hooks)

        self.pydantic_agent = Agent[AgentDeps, DeferredToolRequests | str](
            self.model,
            instructions=self.system_prompt,
            capabilities=capabilities,
            toolsets=self.toolsets,
            deps_type=AgentDeps,
            output_type=(DeferredToolRequests, str),
        )
        self.session.initialized = True

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
    def agent_config(self) -> AgentConfig:
        return self.session.agent_config

    @agent_config.setter
    def agent_config(self, value: AgentConfig):
        self.session.agent_config = value

    @property
    def agent_source(self) -> Literal["static", "dynamic"]:
        return self.session.agent_source

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
        await self.io_adapter.register_host(self)

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
                session_id_fn=lambda: self.run_info.llm_context_id or "",
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
        skills = self.parsed_config.skills
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

        self.default_skill_prompts = []
        for skill_name in default_skills:
            skill = self.parsed_config.skills.get(skill_name)
            if skill:
                try:
                    with open(skill["location"], "r", encoding="utf-8") as f:
                        self.default_skill_prompts.append(f.read())
                except Exception as e:
                    logger.warning(f"Failed to read default skill {skill_name}: {e}")
            else:
                logger.warning(
                    f"Default skill {skill_name} not found in available skills"
                )

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
    def system_prompt(self):
        async def _wrapper() -> str:
            prompt = self.raw_system_prompt
            if self.additional_prompt:
                prompt += f"\n{self.additional_prompt}"

            if hasattr(self, "default_skill_prompts") and self.default_skill_prompts:
                prompt += "\n\n" + "\n\n".join(self.default_skill_prompts)

            if self.skill_catalog:
                prompt += f"\n\n{self.skill_catalog}"

            slot_prompts = await self.invoke_slot(SYSTEM_PROMPT)
            if slot_prompts:
                prompt += "\n\n" + "\n\n".join(slot_prompts)

            return utils.render_template(prompt, agent=self)

        return _wrapper

    async def _wrap_model_request(
        self,
        ctx: RunContext[AgentDeps],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        self.message_history_fallback = list(ctx.messages)
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
                    message_history=message_history,
                    deps=AgentDeps(agent_io=self.agent_io, agent=self),
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

        self.session.record_step(
            result.new_messages(),
        )
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
        if result and isinstance(result.output, DeferredToolRequests):
            return (
                f"Sub-agent {self.name} requested deferred tools, "
                "which is not supported in delegation yet."
            )
        return None


def create_agent(
    name: str,
    parsed_config: Config,
    io_adapter: Any,
    session: AgentSession | None = None,
    parent_session: AgentSession | None = None,
    agent_config: AgentConfig | None = None,
    agent_source: Literal["static", "dynamic"] = "static",
    workspace: Path | str | None = None,
    agent_cls: type | None = None,
) -> LLMBaseAgent:
    if not agent_config:
        agent_config = parsed_config.agent.get(name)
        if not agent_config:
            raise ValueError(f"Agent config for '{name}' not found")

    if not agent_cls:
        agent_type = agent_config.type
        try:
            agent_cls = utils.import_class(agent_type, group="arox.agents")
        except ValueError:
            raise ValueError(f"Unknown agent type: {agent_type} for agent {name}")

    if session is None:
        session = AgentSession(
            path=[*parent_session.path, str(uuid.uuid4())]
            if parent_session
            else [str(uuid.uuid4())],
            agent_name=name,
            agent_config=agent_config.model_copy(deep=True),
            agent_source=agent_source,
            workspace=str(Path(workspace).absolute()) if workspace else None,
            run_info=AgentRunInfo(llm_context_id=str(uuid.uuid4())),
        )
        if parent_session:
            session.owner = parent_session
            session.manager = parent_session.manager
            parent_session.children.append(session.id)

    agent = agent_cls(
        parsed_config=parsed_config,
        io_adapter=io_adapter,
        session=session,
    )
    return agent
