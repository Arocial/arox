from typing import Any


class Plugin:
    def __init__(self, agent):
        self.agent = agent

    def commands(self) -> list:
        """Return a list of Command instances."""
        return []

    def tools(self) -> list[dict[str, Any]]:
        """Return a list of dicts containing 'func' and other kwargs for add_local_tool."""
        return []
