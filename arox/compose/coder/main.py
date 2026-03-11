import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from pydantic_ai import FunctionToolset

from arox import agent_patterns, commands, config
from arox.agent_patterns.chat import ChatAgent
from arox.compose.coder.state import CoderState
from arox.compose.git_commit import GitCommitAgent
from arox.config import TomlConfigParser
from arox.tools.shell import Shell
from arox.ui.io import IOChannel

logger = logging.getLogger(__name__)


class CoderComposer:
    def __init__(self, io_adapter_func):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dump-default-config",
            help="Dump default config to specified file and exit.",
            default="",
        )
        args, unknown_args = parser.parse_known_args()
        cli_configs = config.parse_dot_config(unknown_args)

        default_agent_config = Path(__file__).parent / "config.toml"
        toml_parser = TomlConfigParser(
            config_files=[default_agent_config], override_configs=cli_configs
        )

        self.name = "coder"
        toml_parser.add_argument_group(name=f"composer.{self.name}", expose_raw=True)
        self.config = toml_parser.parse_args()
        getattr(self.config.composer, self.name)

        agent_patterns.init(toml_parser)

        self.git_io_channel = IOChannel()
        self.git_adapter = io_adapter_func(self.git_io_channel)

        git_commit_agent = GitCommitAgent(
            "git_commit_agent",
            toml_parser,
            agent_io=self.git_io_channel,
        )
        self.commit_agent = git_commit_agent

        self.coder_io_channel = IOChannel()
        local_toolset = FunctionToolset()
        coder_agent = ChatAgent(
            "coder",
            toml_parser,
            local_toolset=local_toolset,
            state_cls=CoderState,
            context={"commit_agent": self.commit_agent},
            agent_io=self.coder_io_channel,
        )
        shell_tool = Shell(coder_agent.workspace.absolute())
        shell_tool.register_tool(coder_agent)
        coder_commands = [
            commands.ProjectCommand(coder_agent),
            commands.ModelCommand(coder_agent),
            commands.InvokeToolCommand(coder_agent),
            commands.ListToolCommand(coder_agent),
            commands.ResetCommand(coder_agent),
            commands.InfoCommand(coder_agent),
            commands.CommitCommand(coder_agent),
        ]
        coder_agent.register_commands(coder_commands)

        self.coder_adapter = io_adapter_func(self.coder_io_channel)
        self.coder_adapter.setup(coder_agent)

        self.coder_agent = coder_agent

        # Add commit hooks
        async def pre_step_hook(agent, input_content: str):
            logger.info("Running pre-LLM commit hook")
            await self.commit_agent.auto_commit_changes()

        self.coder_agent.add_pre_step_hook(pre_step_hook)

        if args.dump_default_config:
            logger.debug(f"Dumping default config to {args.dump_default_config}")
            with open(args.dump_default_config, "w") as f:
                toml_parser.dump_default_config(f)
            sys.exit(0)

    async def run(self):
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self.coder_io_channel)
            await stack.enter_async_context(self.git_io_channel)
            await stack.enter_async_context(self.coder_agent)
            await stack.enter_async_context(self.commit_agent)

            asyncio.create_task(self.git_adapter.start())
            asyncio.create_task(self.coder_adapter.start())

            await self.commit_agent.show_agent_info()
            await self.coder_agent.show_agent_info()
            await self.coder_agent.start()


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--ui",
        choices=["textui", "restapi"],
        default="textui",
        help="UI interface to use (textui or restapi)",
    )
    args, _ = parser.parse_known_args()

    if args.ui == "restapi":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        log_dir = Path(".arox")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_dir / "agents.log",
            filemode="a",
        )

    if args.ui == "textui":
        from arox.ui.io import TextIOAdapter

        composer = CoderComposer(TextIOAdapter)
        asyncio.run(composer.run())
    elif args.ui == "restapi":
        from arox.compose.coder.rest_api import CoderRestUI

        app = CoderRestUI()
        app.run()


if __name__ == "__main__":
    main()
