import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Disable fastmcp custom logging
os.environ["FASTMCP_LOG_ENABLED"] = "false"

from arox.core.app import app_setup
from arox.core.llm_base import MainAgent, create_agent

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
        from platformdirs import user_log_dir

        log_dir = Path(user_log_dir("arox"))
        log_dir.mkdir(parents=True, exist_ok=True)
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

    parsed_config = app_setup(
        app_name="chat", profile=profile_name, cli_args=unknown_args
    )

    if args.ui == "vercel_ai":
        from arox.ui.vercel_ai import VercelStreamServer

        server = VercelStreamServer(
            app_name="chat",
            profile=profile_name,
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

        async def run_all():
            from arox.core.session import AgentSession, FileSessionStore, SessionManager

            session_store = FileSessionStore(
                max_age_days=parsed_config.app.session_max_age_days
            )
            session_manager = SessionManager(session_store)
            session_manager.register_session_type(AgentSession)

            await session_store.cleanup()
            session = None
            if args.session:
                session = await session_store.load_session([args.session])
                if not session or not isinstance(session, AgentSession):
                    print(
                        f"Session {args.session} not found or invalid.", file=sys.stderr
                    )
                    sys.exit(1)
                parsed_config.app.main_agent = session.agent_name
                parsed_config.agent[session.agent_name] = session.agent_config

            main_agent = create_agent(
                name=parsed_config.app.main_agent,
                parsed_config=parsed_config,
                io_adapter=io_adapter,
                session=session,
            )
            if not isinstance(main_agent, MainAgent):
                raise TypeError(
                    f"Main agent '{parsed_config.app.main_agent}' must be a MainAgent"
                )
            main_agent.session.manager = session_manager

            async with session_manager, io_adapter:
                async with main_agent:
                    if args.session:
                        await main_agent.agent_io.send(
                            f"Session restored: {args.session}"
                        )
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
