from arox.codebase.file_edit import FileEdit
from arox.codebase.project import ProjectManager
from arox.commands import ProjectCommand
from arox.plugins import Plugin


class ProjectPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        self.project_manager = ProjectManager(agent)
        if hasattr(agent, "state"):
            agent.state.project_manager = self.project_manager
        self.file_edit = FileEdit()

    def commands(self):
        return [ProjectCommand(self.agent)]

    def tools(self):
        return [
            {"func": self.project_manager.read},
            {"func": self.file_edit.replace_in_file, "sequential": True},
            {"func": self.file_edit.write_to_file, "sequential": True},
        ]
