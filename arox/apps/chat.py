import argparse
import asyncio
import logging
import os
from pathlib import Path

# Disable fastmcp custom logging
os.environ["FASTMCP_LOG_ENABLED"] = "false"

from arox.core.composer import Composer

logger = logging.getLogger(__name__)


def run_app(app_name: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui",
        choices=["text", "vercel_ai", "telegram", "feishu"],
        default="text",
        help="UI interface to use (text, vercel_ai, telegram, or feishu)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID to restore a previous session",
    )
    args, unknown_args = parser.parse_known_args()

    unknown_args.append(f"composer.{app_name}.io_adapter={args.ui}")

    if args.ui == "text":
        log_dir = Path(".arox")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_dir / "agents.log",
            filemode="a",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    default_agent_config = Path(__file__).parent / app_name / "config.toml"
    from arox.core.app import app_setup

    app_setup(config_files=[default_agent_config], cli_args=unknown_args)

    if args.ui == "vercel_ai":
        from arox.ui.vercel_ai import VercelStreamServer

        server = VercelStreamServer(
            composer_name=app_name,
            config_files=[default_agent_config],
            cli_args=unknown_args,
        )
        asyncio.run(server.run())
    else:
        composer = Composer(
            app_name,
            session_id=args.session,
            config_files=[default_agent_config],
            cli_args=unknown_args,
        )
        asyncio.run(composer.run())


def run_coder():
    run_app("coder")


def run_general():
    run_app("general")
