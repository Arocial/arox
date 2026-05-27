import logging
import os
from pathlib import Path
from typing import Any

import logfire
from pydantic_ai import FunctionToolset

from arox.core.config import Config, ObservabilityConfig, load_config
from arox.core.llm_base import AgentDeps, MainAgent
from arox.utils import import_class

logger = logging.getLogger(__name__)


def app_setup(
    config_files: list[str | Path] | None = None,
    cli_args: list[str] | dict[str, Any] | None = None,
) -> Config:
    config = load_config(config_files, cli_args)

    for var_name, value in config.app.env_vars.items():
        os.environ[var_name] = value

    for provider, api_key in config.app.api_keys.items():
        provider = provider.upper()
        os.environ[f"{provider}_API_KEY"] = api_key

    setup_llm_observability(config.app.observability)

    return config


def create_main_agent(
    parsed_config: Config,
    io_adapter: Any,
    session_id: str | None = None,
    workspace: Path | str | None = None,
) -> MainAgent:
    main_agent_name = parsed_config.app.main_agent
    agent_config = parsed_config.agent.get(main_agent_name)
    if not agent_config:
        raise ValueError(f"Agent config for '{main_agent_name}' not found")

    main_agent_type = agent_config.type
    try:
        main_agent_cls = import_class(main_agent_type, group="arox.agents")
    except ValueError:
        raise ValueError(
            f"Unknown agent type: {main_agent_type} for main agent {main_agent_name}"
        )

    local_toolset = FunctionToolset[AgentDeps]()

    main_agent = main_agent_cls(
        main_agent_name,
        parsed_config,
        io_adapter=io_adapter,
        local_toolset=local_toolset,
        workspace=workspace,
    )

    main_agent.session_id = session_id

    # Load hooks
    pre_step_hooks = agent_config.pre_step_hooks
    for hook_path in pre_step_hooks:
        hook_func = import_class(hook_path, group="arox.hooks")
        main_agent.add_pre_step_hook(hook_func)

    post_step_hooks = agent_config.post_step_hooks
    for hook_path in post_step_hooks:
        hook_func = import_class(hook_path, group="arox.hooks")
        main_agent.add_post_step_hook(hook_func)

    if not isinstance(main_agent, MainAgent):
        raise TypeError(f"Main agent '{main_agent_name}' must be a MainAgent")

    return main_agent


# Observability & Logging
def setup_llm_observability(ob_config: ObservabilityConfig):
    if ob_config.enable:
        logfire.configure(
            console=False,
            send_to_logfire=ob_config.logfire,
            scrubbing=ob_config.scrubbing,
        )
        # https://github.com/orgs/langfuse/discussions/5036#discussioncomment-15019422
        logfire.instrument_pydantic_ai(version=1)
