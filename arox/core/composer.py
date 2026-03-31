from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import FunctionToolset

from arox.core.config import ComposerConfig
from arox.core.llm_base import AgentDeps
from arox.core.session import AppSession, FileSessionStore, SessionStore
from arox.ui.io import IOChannel
from arox.utils import import_class

if TYPE_CHECKING:
    from arox.core.llm_base import LLMBaseAgent

logger = logging.getLogger(__name__)


class Composer:
    def __init__(
        self,
        name: str,
        workspace: Path | str | None = None,
        session_id: str | None = None,
        config_files: list[str | Path] | None = None,
        cli_args: list[str] | dict[str, Any] | None = None,
        session_store: SessionStore | None = None,
    ):
        self.name = name
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.session_id = session_id

        from arox.core.config import load_config

        self.parsed_config, self.config_dirs = load_config(
            config_files, cli_args, self.workspace
        )

        self.session_store: SessionStore = session_store or FileSessionStore()
        self.session = AppSession.create(self.name)

        composer_config = self.parsed_config.composer.get(name)
        if not composer_config:
            raise ValueError(f"Composer config for '{name}' not found")
        self.composer_config: ComposerConfig = composer_config

        io_adapter_cls = import_class(
            self.composer_config.io_adapter, group="arox.io_adapters"
        )
        self.io_adapter = io_adapter_cls()

        self.subagents = {}
        self.io_channels = {}

        self._init_agents()

    def _load_agent_hooks(self, agent, agent_config):
        pre_step_hooks = agent_config.pre_step_hooks
        for hook_path in pre_step_hooks:
            hook_func = import_class(hook_path, group="arox.hooks")
            agent.add_pre_step_hook(hook_func)

        post_step_hooks = agent_config.post_step_hooks
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
            agent_config = self.parsed_config.agent.get(agent_name)
            if not agent_config:
                raise ValueError(f"Agent config for '{agent_name}' not found")
            agent_configs[agent_name] = agent_config

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
                self.parsed_config,
                agent_io=self.io_channels[agent_name],
                workspace=self.workspace,
                config_dirs=self.config_dirs,
            )
            self._load_agent_hooks(agent, agent_configs[agent_name])
            self.subagents[agent_name] = agent

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
            self.parsed_config,
            local_toolset=local_toolset,
            agent_io=self.io_channels[main_agent_name],
            workspace=self.workspace,
            config_dirs=self.config_dirs,
        )

        from arox.plugins.capabilities import SUBAGENT

        def get_subagent(name: str):
            return self.subagents.get(name)

        main_agent.provide_capability(SUBAGENT, get_subagent)

        self._load_agent_hooks(main_agent, agent_configs[main_agent_name])
        self.main_agent = main_agent

        self.io_adapter.setup(main_agent)

    def _all_agents(self) -> dict[str, LLMBaseAgent]:
        agents = dict(self.subagents)
        if self.main_agent:
            agents[self.main_agent.name] = self.main_agent
        return agents

    async def _init_session(self, session_id: str | None = None):
        restored = False
        if session_id:
            loaded = await self.session_store.load_session(session_id)
            if loaded:
                self.session = loaded
                restored = True
                await self.main_agent.agent_io.agent_send(
                    f"Session restored: {self.session.id}"
                )

        if not restored:
            self.session = AppSession.create(self.name)

        for name, agent in self._all_agents().items():
            agent.restore_session(self.session.get_agent_session(name))

    async def _save_session(self):
        await self.session_store.save_session(self.session)

    async def run(self):
        if self.main_agent is None:
            raise RuntimeError("Main agent is not initialized")

        async with contextlib.AsyncExitStack() as stack:
            for io_channel in self.io_channels.values():
                await stack.enter_async_context(io_channel)

            for agent in self.subagents.values():
                await stack.enter_async_context(agent)
            await stack.enter_async_context(self.main_agent)

            asyncio.create_task(self.io_adapter.start())

            await self._init_session(self.session_id)

            for agent in self.subagents.values():
                await agent.show_agent_info()
            await self.main_agent.show_agent_info()

            try:
                if hasattr(self.main_agent, "start"):
                    await self.main_agent.start()
                else:
                    logger.error("Main agent does not have a start method")
            finally:
                await self._save_session()
