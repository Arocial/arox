from __future__ import annotations

import contextlib
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import FunctionToolset

from arox.core.config import ComposerConfig
from arox.core.io import AbstractIOAdapter
from arox.core.llm_base import AgentDeps, DelegatableAgent, MainAgent
from arox.core.session import ComposerSession, FileSessionStore, SessionStore
from arox.utils import import_class

if TYPE_CHECKING:
    from arox.core.llm_base import LLMBaseAgent

logger = logging.getLogger(__name__)


class Composer:
    def __init__(
        self,
        name: str,
        io_adapter: AbstractIOAdapter,
        workspace: Path | str | None = None,
        session_id: str | None = None,
        config_files: list[str | Path] | None = None,
        cli_args: list[str] | dict[str, Any] | None = None,
        session_store: SessionStore | None = None,
    ):
        self.name = name
        self.io_adapter = io_adapter
        self.workspace = Path(workspace).absolute() if workspace else Path.cwd()
        self.session_id = session_id
        self.id = str(uuid.uuid4())

        from arox.core.config import load_config

        self.parsed_config = load_config(config_files, cli_args, self.workspace)

        self.session_store: SessionStore = session_store or FileSessionStore(
            max_age_days=self.parsed_config.app.session_max_age_days
        )
        self.session = ComposerSession.create(self.name, workspace=str(self.workspace))

        composer_config = self.parsed_config.composer.get(name)
        if not composer_config:
            raise ValueError(f"Composer config for '{name}' not found")
        self.composer_config: ComposerConfig = composer_config

        self.subagents = {}

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
                io_adapter=self.io_adapter,
                workspace=self.workspace,
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
            io_adapter=self.io_adapter,
            local_toolset=local_toolset,
            workspace=self.workspace,
        )

        from arox.plugins.capabilities import FORK_SESSION, SUBAGENT

        def get_subagent(name: str):
            return self.subagents.get(name)

        main_agent.provide_capability(SUBAGENT, get_subagent)

        async def _fork(agent_name: str, event_index: int) -> str:
            return await self.fork_session(agent_name, event_index)

        main_agent.provide_capability(FORK_SESSION, _fork)

        exposed_subagents = {
            name: agent
            for name, agent in self.subagents.items()
            if isinstance(agent, DelegatableAgent)
        }

        if exposed_subagents:
            subagent_descriptions = "\n".join(
                f"- {name}: {agent_configs[name].description or 'No description'}"
                for name in exposed_subagents
            )

            async def delegate_to_subagent(subagent_name: str, task: str) -> str:
                f"""Delegate a task to a specific subagent.

                Available subagents:
                {subagent_descriptions}
                """
                agent = exposed_subagents.get(subagent_name)
                if not agent:
                    return f"Error: Subagent '{subagent_name}' not found. Available subagents: {', '.join(exposed_subagents.keys())}"

                main_agent.agent_session.add_event(
                    "subagent_call",
                    {"subagent": agent.name, "task": task},
                )
                result = await agent.run_task(task)
                return result or "Task completed with no output."

            main_agent.add_local_tool(delegate_to_subagent)

        if not isinstance(main_agent, MainAgent):
            raise TypeError(f"Main agent '{main_agent_name}' must be a MainAgent")

        self._load_agent_hooks(main_agent, agent_configs[main_agent_name])
        self.main_agent = main_agent

    def all_agents(self) -> dict[str, LLMBaseAgent]:
        agents = dict(self.subagents)
        if self.main_agent:
            agents[self.main_agent.name] = self.main_agent
        return agents

    async def _init_session(self, session_id: str | None = None):
        await self.session_store.cleanup()

        restored = False
        if session_id:
            loaded = await self.session_store.load_session(session_id)
            if loaded:
                self.session = loaded
                restored = True
                await self.main_agent.agent_io.send(
                    f"Session restored: {self.session.id}"
                )

        if not restored:
            self.session = ComposerSession.create(
                self.name, workspace=str(self.workspace)
            )

        for name, agent in self.all_agents().items():
            agent.restore_session(self.session.get_agent_session(name))

    async def fork_session(self, agent_name: str, event_index: int) -> str:
        """Fork the current session at ``(agent_name, event_index)``.

        Persists both the current session (with its in-memory state) and
        the new branch. The new branch is *not* swapped in — the user
        resumes it via ``--resume <new_id>``. Returns the new session id.
        """
        new_session = self.session.fork_at(agent_name, event_index)
        await self._save_session()
        await self.session_store.save_session(new_session)
        return new_session.id

    async def _save_session(self):
        last_user_messages = []
        if self.main_agent and hasattr(self.main_agent, "message_history"):
            from pydantic_ai.messages import ModelRequest, UserPromptPart

            for msg in reversed(self.main_agent.message_history):
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, UserPromptPart):
                            content = part.content
                            if isinstance(content, str):
                                last_user_messages.append(content)
                            elif isinstance(content, (list, tuple)):
                                text_parts = [c for c in content if isinstance(c, str)]
                                if text_parts:
                                    last_user_messages.append(" ".join(text_parts))
                            if len(last_user_messages) >= 2:
                                break
                if len(last_user_messages) >= 2:
                    break

        if not last_user_messages:
            return

        self.session.metadata["last_user_messages"] = list(reversed(last_user_messages))

        await self.session_store.save_session(self.session)

    async def run(self):
        async with contextlib.AsyncExitStack() as stack:
            await self.io_adapter.register_composer(self)

            for agent in self.subagents.values():
                await stack.enter_async_context(agent)
            await stack.enter_async_context(self.main_agent)

            await self._init_session(self.session_id)

            try:
                for agent in self.subagents.values():
                    await agent.show_agent_info()
                await self.main_agent.show_agent_info()

                await self.main_agent.run()
            finally:
                await self._save_session()
