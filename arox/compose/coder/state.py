import logging

from arox.agent_patterns.state import SimpleState
from arox.codebase import project

logger = logging.getLogger(__name__)


class CoderState(SimpleState):
    def __init__(
        self,
        agent,
    ):
        super().__init__(agent)
        self.project_manager = project.ProjectManager(self.workspace, agent)
        self.chat_files.set_candidate_generator(self.project_manager.get_tracked_files)
