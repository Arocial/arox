import asyncio

from arox.commands import CommandCompleter
from arox.compose.coder.main import CoderComposer
from arox.ui import TUIByIO
from arox.ui.io import TextIOAdapter


class CoderTUI(TUIByIO):
    def on_mount(self) -> None:
        composer = CoderComposer(lambda: self.io_adapter)
        self.input_suggester = CommandCompleter(
            composer.coder_agent.command_manager
        ).textual_suggester
        self.run_worker(composer.run, name="composer", exclusive=True)


class CoderTextUI:
    def run(self):
        composer = CoderComposer(TextIOAdapter.factory())
        asyncio.run(composer.run())
