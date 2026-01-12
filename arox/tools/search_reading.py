from arox.agent_patterns.state import SimpleState


class SearchReading:
    def __init__(self, state: SimpleState):
        self.agent_state = state

    def add_files(self, paths: list[str]):
        """
        Add files to the chat context.

        Note:
          The content of added files will be provided in next user message,
          not the return value of the tool.

        Args:
            paths: **List** of paths to files to be added to chat context.
            e.g. ["a_file", "b_file"].

        Returns:
            str: Success message or error description
        """
        try:
            chat_files = self.agent_state.chat_files

            if not isinstance(paths, list):
                paths = [paths]
            results = chat_files.add_by_names(paths)
            msg = []
            if results["succeed"]:
                msg.append(
                    f"Successfully added files: {' '.join(results['succeed'])}. See <files> section below."
                )
            if results["not_exist"]:
                msg.append(f"File not exist: {' '.join(results['not_exist'])}")

            return "\n".join(msg)

        except Exception as e:
            return f"Error reading files: {str(e)}"
