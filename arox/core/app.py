import logging
import os
from pathlib import Path
from typing import Any

import logfire

from arox.core.config import Config, ObservabilityConfig, load_config

logger = logging.getLogger(__name__)

# Capture the environment before app_setup modifies it
_ORIGINAL_ENV = os.environ.copy()


def get_original_env_copy() -> dict[str, str]:
    """Return a copy of the environment as it was before app_setup."""
    return _ORIGINAL_ENV.copy()


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
