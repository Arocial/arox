import logging
import os
import sys
from pathlib import Path
from typing import Any

import logfire

from arox.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def app_init(
    config_files: list[str | Path] | None = None,
    cli_args: list[str] | dict[str, Any] | None = None,
) -> AppConfig:
    config = load_config(config_files, cli_args)
    if config.dump_config:
        logger.debug(f"Dumping default config to {config.dump_config}")
        with open(config.dump_config, "w") as f:
            f.write(config.model_dump_json(indent=2))
        sys.exit(0)

    setup_llm_observability(config)

    for provider, api_key in config.api_keys.items():
        provider = provider.upper()
        os.environ[f"{provider}_API_KEY"] = api_key

    for var_name, value in config.env_vars.items():
        os.environ[var_name] = value

    return config


# Observability & Logging
def setup_llm_observability(config: AppConfig):
    if config.observability.enable:
        logfire.configure(
            console=False,
            send_to_logfire=config.observability.logfire,
            scrubbing=config.observability.scrubbing,
        )
        # https://github.com/orgs/langfuse/discussions/5036#discussioncomment-15019422
        logfire.instrument_pydantic_ai(version=1)
