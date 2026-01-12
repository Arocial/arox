import logging
import os
from pathlib import Path

import logfire

logger = logging.getLogger(__name__)


def init(config_parser):
    conf = add_agent_options(config_parser)
    setup_llm_observability(conf)
    return conf


# Observability & Logging
def setup_llm_observability(conf):
    if conf.observability.enable:
        logfire.configure(
            send_to_logfire=conf.observability.logfire,
            scrubbing=conf.observability.scrubbing,
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)


def add_agent_options(parser):
    parser.add_argument(
        "model_ref",
        default="deepseek/deepseek-chat",
        help="Model to use with ChatLiteLLM",
    )
    parser.add_argument(
        "workspace",
        default="workspace",
        help="Path for agents to work, default to $current_dir/workspace",
    )

    # Observability configuration group
    obs_group = parser.add_argument_group(
        name="observability", help="Observability Configuration"
    )
    obs_group.add_argument(
        "enable",
        default=False,
        help="Enable observation",
    )
    obs_group.add_argument(
        "scrubbing",
        default=None,
        help="Enable ",
    )
    obs_group.add_argument(
        "logfire",
        default=False,
        help="Use logfire backend",
    )

    # API Keys group
    parser.add_argument_group("api_keys", "API Keys", expose_raw=True)
    parser.add_argument_group("env_vars", "Environment variables", expose_raw=True)
    # MCP Servers group
    parser.add_argument_group(
        "mcp_servers", "MCP Server Configurations", expose_raw=True
    )

    args = parser.parse_args()

    for provider, api_key in args.api_keys.items():
        provider = provider.upper()
        os.environ[f"{provider}_API_KEY"] = api_key

    for var_name, value in args.env_vars.items():
        os.environ[var_name] = value

    # Ensure workspace directory exists
    workspace_path = Path(args.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    add_extra_config(args)
    return args


def add_extra_config(args):
    args.user_path = Path.home() / ".arox"
    args.user_path.mkdir(parents=True, exist_ok=True)

    args.verbose_out_path = args.user_path / "__verbose_out__"
    args.verbose_out_path.mkdir(parents=True, exist_ok=True)
