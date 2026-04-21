import argparse
import asyncio
import logging
import os
from pathlib import Path

# Disable fastmcp custom logging
os.environ["FASTMCP_LOG_ENABLED"] = "false"

from arox.core.composer import Composer

logger = logging.getLogger(__name__)


def main():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app",
        default=None,
        help="App name to run (e.g., coder, general)",
    )
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
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (for vercel_ai UI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (for vercel_ai UI)",
    )
    args, unknown_args = parser.parse_known_args()

    app_name = args.app
    if not app_name:
        script_name = Path(sys.argv[0]).name
        if script_name.startswith("arox-"):
            app_name = script_name[5:]
        else:
            app_name = "coder"

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
            host=args.host,
            port=args.port,
        )
        asyncio.run(server.run())
    else:
        if args.ui == "text":
            from arox.ui.text_io import TextIOAdapter

            io_adapter = TextIOAdapter()
        elif args.ui == "telegram":
            from arox.ui.telegram import TelegramIOAdapter

            io_adapter = TelegramIOAdapter()
        elif args.ui == "feishu":
            from arox.ui.feishu import FeishuIOAdapter

            io_adapter = FeishuIOAdapter()
        else:
            raise ValueError(f"Unknown UI: {args.ui}")

        composer = Composer(
            app_name,
            io_adapter=io_adapter,
            session_id=args.session,
            config_files=[default_agent_config],
            cli_args=unknown_args,
        )

        async def run_all():
            async with io_adapter:
                await composer.run()

        asyncio.run(run_all())


if __name__ == "__main__":
    main()
