import argparse
import asyncio
import logging
from pathlib import Path

from arox.agent_patterns.composer import Composer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui",
        choices=["text", "vercel_ai", "telegram", "feishu"],
        default="text",
        help="UI interface to use (text, vercel_ai, telegram, or feishu)",
    )
    args, unknown_args = parser.parse_known_args()

    if args.ui == "text":
        unknown_args.append("composer.coder.io_adapter=arox.ui.text_io.TextIOAdapter")
    elif args.ui == "vercel_ai":
        unknown_args.append(
            "composer.coder.io_adapter=arox.ui.vercel_ai.VercelStreamIOAdapter"
        )
    elif args.ui == "telegram":
        unknown_args.append(
            "composer.coder.io_adapter=arox.ui.telegram.TelegramIOAdapter"
        )
    elif args.ui == "feishu":
        unknown_args.append("composer.coder.io_adapter=arox.ui.feishu.FeishuIOAdapter")
    else:
        raise ValueError(f"Unknown UI: {args.ui}")

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

    default_agent_config = Path(__file__).parent / "config.toml"
    from arox.agent_patterns import app_init

    app_config = app_init(config_files=[default_agent_config], cli_args=unknown_args)
    composer = Composer("coder", app_config)

    asyncio.run(composer.run())


if __name__ == "__main__":
    main()
