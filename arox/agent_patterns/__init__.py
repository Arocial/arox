import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def init(config_parser):
    conf = add_agent_options(config_parser)
    setup_llm_observability(conf)
    return conf


# Observability & Logging
def setup_llm_observability(conf):
    # TODO setup observation
    pass


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
        "provider",
        default=None,
        help="Observability provider to use (default: None)",
    )
    obs_group.add_argument(
        "langfuse_public_key",
        help="Langfuse public key",
    )
    obs_group.add_argument(
        "langfuse_secret_key",
        help="Langfuse secret key",
    )
    obs_group.add_argument("langfuse_host", help="Langfuse host URL")

    # API Keys group
    parser.add_argument_group("api_keys", "API Keys", expose_raw=True)
    parser.add_argument_group("env_vars", "Environment variables", expose_raw=True)
    # MCP Servers group
    parser.add_argument_group(
        "mcp_servers", "MCP Server Configurations", expose_raw=True
    )

    args = parser.parse_args()

    # Set environment variables from config
    if args.observability.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = args.observability.langfuse_public_key
    if args.observability.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = args.observability.langfuse_secret_key
    if args.observability.langfuse_host:
        os.environ["LANGFUSE_HOST"] = args.observability.langfuse_host

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
