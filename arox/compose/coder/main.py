import argparse
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
from arox.utils import run_command

logger = logging.getLogger(__name__)


class CoderComposer:
    def __init__(self, io_adapter_factory):
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
        composer_group = toml_parser.add_argument_group(
            name=f"composer.{self.name}", expose_raw=True
        )
        composer_group.add_argument("pre_commit_cmd", default=None)
        self.config = toml_parser.parse_args()
        composer_config = getattr(self.config.composer, self.name)
        self.pre_commit_cmd = composer_config.pre_commit_cmd

        agent_patterns.init(toml_parser)

        git_commit_agent = GitCommitAgent(
            "git_commit_agent",
            toml_parser,
            io_adapter=io_adapter_factory(),
        )
        self.commit_agent = git_commit_agent

        local_toolset = FunctionToolset()
        coder_agent = ChatAgent(
            "coder",
            toml_parser,
            local_toolset=local_toolset,
            state_cls=CoderState,
            context={"commit_agent": self.commit_agent},
            io_adapter=io_adapter_factory(),
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

        self.coder_agent = coder_agent

        # Add commit hooks
        async def before_llm_hook(agent, input_content: str):
            logger.info("Running pre-LLM commit hook")
            await self.commit_agent.auto_commit_changes()

        async def after_llm_hook(agent, input_content: str):
            logger.info("Running post-LLM commit hook")
            if self.pre_commit_cmd:
                stdout, stderr, returncode = await run_command(self.pre_commit_cmd)
                if returncode != 0:
                    logger.error(f"Pre-commit command failed: {self.pre_commit_cmd}")
                    logger.error(f"stdout: {stdout}")
                    logger.error(f"stderr: {stderr}")

            co_author = f"arox-coder/{agent.provider_model}"
            await self.commit_agent.auto_commit_changes(co_author=co_author)

        self.coder_agent.add_before_step_hook(before_llm_hook)
        self.coder_agent.add_after_step_hook(after_llm_hook)

        if args.dump_default_config:
            logger.debug(f"Dumping default config to {args.dump_default_config}")
            with open(args.dump_default_config, "w") as f:
                toml_parser.dump_default_config(f)
            sys.exit(0)

    async def run(self):
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self.coder_agent)
            await stack.enter_async_context(self.commit_agent)

            await self.commit_agent.show_agent_info()
            await self.coder_agent.show_agent_info()
            await self.coder_agent.start()


def main():
    log_dir = Path(".arox")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_dir / "agents.log",
        filemode="a",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ui",
        choices=["tui", "textui", "restapi"],
        default="textui",
        help="UI interface to use (tui, textui or restapi)",
    )
    args = parser.parse_args()

    if args.ui == "tui":
        from arox.compose.coder.ui import CoderTUI

        app = CoderTUI("Coder")
    elif args.ui == "textui":
        from arox.compose.coder.ui import CoderTextUI

        app = CoderTextUI()
    elif args.ui == "restapi":
        from arox.compose.coder.rest_api import CoderRestUI

        app = CoderRestUI()

    app.run()


if __name__ == "__main__":
    main()
