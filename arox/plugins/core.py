from arox.commands import (
    CommitCommand,
    CompactionCommand,
    InfoCommand,
    InvokeToolCommand,
    ListToolCommand,
    ModelCommand,
    ResetCommand,
)
from arox.plugins import Plugin
from arox.tools import ask_human
from arox.tools.shell import Shell


class CorePlugin(Plugin):
    def commands(self):
        return [
            ModelCommand(self.agent),
            InvokeToolCommand(self.agent),
            ListToolCommand(self.agent),
            InfoCommand(self.agent),
            ResetCommand(self.agent),
            CommitCommand(self.agent),
            CompactionCommand(self.agent),
        ]

    def tools(self):
        from typing import Any

        tools: list[dict[str, Any]] = [{"func": ask_human}]
        shell_tool = Shell(self.agent.workspace.absolute())
        if not shell_tool.disabled:
            tools.append({"func": shell_tool.shell})
        return tools
