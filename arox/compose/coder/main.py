import argparse
import asyncio
import logging
import sys
from pathlib import Path

from arox import agent_patterns, config
from arox.compose.composer import Composer
from arox.config import TomlConfigParser

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui",
        choices=["text", "vercel_ai", "telegram", "feishu"],
        default="text",
        help="UI interface to use (text, vercel_ai, telegram, or feishu)",
    )
    parser.add_argument(
        "--dump-default-config",
        help="Dump default config to specified file and exit.",
        default="",
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

    cli_configs = config.parse_dot_config(unknown_args)

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
    toml_parser = TomlConfigParser(
        config_files=[default_agent_config], override_configs=cli_configs
    )

    agent_patterns.init(toml_parser)

    composer = Composer("coder", toml_parser)

    if args.dump_default_config:
        logger.debug(f"Dumping default config to {args.dump_default_config}")
        with open(args.dump_default_config, "w") as f:
            toml_parser.dump_default_config(f)
        sys.exit(0)

    asyncio.run(composer.run())


if __name__ == "__main__":
    main()
