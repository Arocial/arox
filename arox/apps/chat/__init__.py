import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Disable fastmcp custom logging
os.environ["FASTMCP_LOG_ENABLED"] = "false"

from arox.core.app import app_setup, create_main_agent

logger = logging.getLogger(__name__)


def main(profile: str | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=profile,
        help="Profile to load (e.g., coder, general)",
    )
    parser.add_argument(
        "--ui",
        choices=["text", "vercel_ai", "telegram", "feishu", "headless"],
        default="text",
        help="UI interface to use (text, vercel_ai, telegram, feishu, or headless)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt for headless mode. Combined with stdin if both are provided.",
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

    profile_name = args.profile or "coder"

    if args.ui in ("text", "headless"):
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

    default_agent_config = (
        Path(__file__).parent / "profiles" / profile_name / "config.toml"
    )

    parsed_config = app_setup(
        config_files=[default_agent_config], cli_args=unknown_args
    )

    if args.ui == "vercel_ai":
        from arox.ui.vercel_ai import VercelStreamServer

        server = VercelStreamServer(
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
        elif args.ui == "headless":
            from arox.ui.headless import HeadlessIOAdapter

            prompt = _resolve_headless_prompt(args.prompt)
            if not prompt:
                print(
                    "headless mode requires a prompt via --prompt or stdin",
                    file=sys.stderr,
                )
                sys.exit(2)
            io_adapter = HeadlessIOAdapter(prompt=prompt)
        else:
            raise ValueError(f"Unknown UI: {args.ui}")

        main_agent = create_main_agent(
            parsed_config,
            io_adapter=io_adapter,
            session_id=args.session,
        )

        async def run_all():
            async with io_adapter:
                await io_adapter.register_host(main_agent)
                async with main_agent:
                    await main_agent.show_agent_info()
                    await main_agent.run()

        if args.ui == "headless":
            from arox.ui.headless import HeadlessIOAdapter

            assert isinstance(io_adapter, HeadlessIOAdapter)
            try:
                asyncio.run(run_all())
            except Exception:
                logger.exception("Headless run failed")
                sys.exit(1)
            if io_adapter.error is not None:
                print(f"Headless run failed: {io_adapter.error}", file=sys.stderr)
                sys.exit(1)
        else:
            asyncio.run(run_all())


def _resolve_headless_prompt(cli_prompt: str | None) -> str:
    """Merge --prompt with stdin content (cli first, stdin appended).

    Returns the combined prompt, stripped. If stdin is a tty, it is not
    consumed; only piped/redirected input is read.
    """
    parts: list[str] = []
    if cli_prompt:
        parts.append(cli_prompt)
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text:
            parts.append(stdin_text)
    return "\n".join(parts).strip()


def coder():
    main(profile="coder")


def general():
    main(profile="general")


if __name__ == "__main__":
    main()
