import logging
import os
from pathlib import Path

import logfire

from arox.config import AppConfig

logger = logging.getLogger(__name__)


def init(config: AppConfig):
    setup_llm_observability(config)

    for provider, api_key in config.api_keys.items():
        provider = provider.upper()
        os.environ[f"{provider}_API_KEY"] = api_key

    for var_name, value in config.env_vars.items():
        os.environ[var_name] = value

    add_extra_config(config)
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


def add_extra_config(config: AppConfig):
    config.user_path = Path.home() / ".arox"
    config.user_path.mkdir(parents=True, exist_ok=True)

    config.verbose_out_path = config.user_path / "__verbose_out__"
    config.verbose_out_path.mkdir(parents=True, exist_ok=True)
