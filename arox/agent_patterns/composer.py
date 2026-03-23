import asyncio
import contextlib
import logging

from pydantic_ai import FunctionToolset

from arox.agent_patterns.llm_base import AgentDeps
from arox.config import TomlConfigParser
from arox.ui.io import IOChannel
from arox.utils import import_class

logger = logging.getLogger(__name__)


class Composer:
    def __init__(self, name: str, toml_parser: TomlConfigParser):
        self.name = name
        self.toml_parser = toml_parser

        composer_group = toml_parser.add_argument_group(
            name=f"composer.{self.name}", expose_raw=True
        )
        composer_group.add_argument("main_agent", required=True)
        composer_group.add_argument("subagents", default=[])
        composer_group.add_argument(
            "io_adapter", default="arox.ui.text_io.TextIOAdapter"
        )

        self.config = toml_parser.parse_args()
        self.composer_config = getattr(self.config.composer, self.name)

        io_adapter_cls = import_class(
            self.composer_config.io_adapter, group="arox.io_adapters"
        )
        self.io_adapter = io_adapter_cls()

        self.agents = {}
        self.io_channels = {}

        self._init_agents()

    def _load_agent_hooks(self, agent, agent_config):
        pre_step_hooks = getattr(agent_config, "pre_step_hooks", [])
        for hook_path in pre_step_hooks:
            hook_func = import_class(hook_path, group="arox.hooks")
            agent.add_pre_step_hook(hook_func)

        post_step_hooks = getattr(agent_config, "post_step_hooks", [])
        for hook_path in post_step_hooks:
            hook_func = import_class(hook_path, group="arox.hooks")
            agent.add_post_step_hook(hook_func)

    def _init_agents(self):
        main_agent_name = self.composer_config.main_agent
        subagent_names = self.composer_config.subagents

        all_agent_names = [main_agent_name] + subagent_names

        # First pass: create IO channels and parse agent configs to get their types
        agent_configs = {}
        for agent_name in all_agent_names:
            agent_group = self.toml_parser.add_argument_group(
                name=f"agent.{agent_name}", expose_raw=True
            )
            agent_group.add_argument("type", default="chat")
            self.config = self.toml_parser.parse_args()
            agent_configs[agent_name] = getattr(self.config.agent, agent_name)

            io_channel = IOChannel()
            self.io_channels[agent_name] = io_channel
            self.io_adapter.add_adapter_io(io_channel)

        # Second pass: instantiate subagents
        for agent_name in subagent_names:
            agent_type = agent_configs[agent_name].type
            try:
                agent_cls = import_class(agent_type, group="arox.agents")
            except ValueError:
                raise ValueError(
                    f"Unknown agent type: {agent_type} for agent {agent_name}"
                )

            agent = agent_cls(
                agent_name,
                self.toml_parser,
                agent_io=self.io_channels[agent_name],
            )
            self._load_agent_hooks(agent, agent_configs[agent_name])
            self.agents[agent_name] = agent

        # Third pass: instantiate main agent with context of subagents
        main_agent_type = agent_configs[main_agent_name].type
        try:
            main_agent_cls = import_class(main_agent_type, group="arox.agents")
        except ValueError:
            raise ValueError(
                f"Unknown agent type: {main_agent_type} for main agent {main_agent_name}"
            )

        local_toolset = FunctionToolset[AgentDeps]()

        main_agent = main_agent_cls(
            main_agent_name,
            self.toml_parser,
            local_toolset=local_toolset,
            agent_io=self.io_channels[main_agent_name],
        )

        for name, agent in self.agents.items():
            if name != main_agent_name:
                main_agent.register_dependency(name, agent)

        self._load_agent_hooks(main_agent, agent_configs[main_agent_name])
        self.agents[main_agent_name] = main_agent
        self.main_agent = main_agent

        self.io_adapter.setup(main_agent)

    async def run(self):
        async with contextlib.AsyncExitStack() as stack:
            for io_channel in self.io_channels.values():
                await stack.enter_async_context(io_channel)

            for agent in self.agents.values():
                await stack.enter_async_context(agent)

            asyncio.create_task(self.io_adapter.start())

            for agent in self.agents.values():
                if agent != self.main_agent:
                    await agent.show_agent_info()
            await self.main_agent.show_agent_info()

            if hasattr(self.main_agent, "start"):
                await self.main_agent.start()
            else:
                logger.error("Main agent does not have a start method")
