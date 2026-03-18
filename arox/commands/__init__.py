import json
import logging

import yaml
from prompt_toolkit.completion import Completer, Completion

logger = logging.getLogger(__name__)


def parse_cmdline(cmdline):
    if not cmdline.startswith("/"):
        return None, None

    cmd = cmdline.split(" ", 1)
    c_name = cmd[0][1:]
    c_arg = cmd[1] if len(cmd) > 1 else None
    return c_name, c_arg


class CommandCompleter(Completer):
    """Main completer that delegates to specific command completers"""

    def __init__(self, manager):
        self.command_manager = manager

    def get_completions(self, document, complete_event):
        yield from self._get_completions(document.text)

    def _get_completions(self, text):
        name, args = parse_cmdline(text)
        if not name:
            return
        if args is None:  # Complete command names
            candidates = self.command_manager.command_names()
            for candidate in candidates:
                if name in candidate:
                    yield Completion(
                        candidate, start_position=-len(name), display=candidate
                    )
            return

        yield from self.command_manager.get_completions(name, args)


class Command:
    """Base class for agent commands"""

    command: str = ""
    description: str = ""

    def __init__(self, agent):
        self.agent = agent

    def slashes(self) -> list[str]:
        return [self.command]

    async def execute(self, name: str, arg: str):
        """Execute command with given input"""
        raise NotImplementedError

    def get_completions(self, name, args):
        yield from []


class ProjectCommand(Command):
    description = "Add files to context - /add <file1> [file2...] /add_file_list"

    def slashes(self) -> list[str]:
        return ["add", "add_file_list"]

    async def execute(self, name: str, arg: str):
        project_manager = self.agent.state.project_manager
        if name == "add":
            files = arg.split() if arg else []
            if not files:
                await self.agent.agent_io.agent_send("Please specify files.")
                return
            await project_manager.read_by_user(files)
        elif name == "add_file_list":
            project_manager.add_project_files()

    def get_completions(self, name, args):
        # Parse the arguments to get the current word being completed
        if not args:
            current_word = ""
        else:
            parts = args.split()
            if args.endswith(" "):
                current_word = ""
            else:
                current_word = parts[-1] if parts else ""

        if name == "add":
            candidates = self.agent.state.project_manager.candidates()
        else:
            candidates = []

        # Filter candidates based on current word
        for candidate in candidates:
            if current_word in candidate:
                yield Completion(
                    candidate, start_position=-len(current_word), display=candidate
                )


class ModelCommand(Command):
    command = "model"
    description = "Switch LLM model - /model <model_name>"

    async def execute(self, name: str, arg: str):
        if not arg:
            await self.agent.agent_io.agent_send("Please specify a model name")
            return
        self.agent.set_model(arg)


class InvokeToolCommand(Command):
    command = "invoke-tool"
    description = "Invoke a registered tool - /invoke-tool <function_name> [json_args]"

    async def execute(self, name: str, arg: str):
        tool_registry = self.agent.tool_registry

        parts = arg.split(maxsplit=1)
        if len(parts) < 1:
            await self.agent.agent_io.agent_send(
                "Usage: /invoke-tool <function_name> [json_args]"
            )
            return

        function_name = parts[0]
        args_str = parts[0] if len(parts) > 1 else "{}"

        try:
            args = json.loads(args_str)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object (dictionary).")
        except json.JSONDecodeError as e:
            await self.agent.agent_io.agent_send(f"Error: Invalid JSON arguments: {e}")
            return
        except ValueError as e:
            await self.agent.agent_io.agent_send(f"Error: {e}")
            return

        # Prepare the tool_call structure expected by execute_tool_call
        # Note: We don't have a real tool_call ID here, as it's a direct invocation.
        tool_call_data = {
            "id": f"cmd_{function_name}",  # Generate a placeholder ID
            "type": "function",
            "function": {"name": function_name, "arguments": json.dumps(args)},
        }

        try:
            await self.agent.agent_io.agent_send(
                f"Invoking tool '{function_name}' with args: {args}"
            )
            result = await tool_registry.execute_tool_call(tool_call_data)
            await self.agent.agent_io.agent_send(
                f"Tool '{function_name}' executed successfully."
            )
            await self.agent.agent_io.agent_send("Result:")
            await self.agent.agent_io.agent_send(result)
        except ValueError as e:
            await self.agent.agent_io.agent_send(
                f"Error invoking tool '{function_name}': {e}"
            )
        except ConnectionError as e:
            await self.agent.agent_io.agent_send(
                f"Error connecting to MCP server for tool '{function_name}': {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error invoking tool '{function_name}': {e}", exc_info=True
            )
            await self.agent.agent_io.agent_send(f"An unexpected error occurred: {e}")


class ListToolCommand(Command):
    command = "list-tools"
    description = "List all registered tools"

    async def execute(self, name: str, arg: str):
        tool_registry = self.agent.tool_registry
        tool_specs = await tool_registry.get_tools_specs()
        if not tool_specs:
            await self.agent.agent_io.agent_send("No tools registered.")
            return

        await self.agent.agent_io.agent_send("Registered Tools:")
        await self.agent.agent_io.agent_send(yaml.safe_dump(tool_specs))


class InfoCommand(Command):
    command = "info"
    description = "Show current chat files and model in use - /info"

    async def execute(self, name: str, arg: str):
        # Show current model
        current_model = getattr(self.agent, "provider_model", "Unknown")
        await self.agent.agent_io.agent_send(f"Current model: {current_model}")

        # Show chat files
        session_files = self.agent.state.project_manager.session_files
        if session_files:
            await self.agent.agent_io.agent_send(
                f"\nChat files ({len(session_files)}):"
            )
            for file_path in session_files:
                await self.agent.agent_io.agent_send(f"  - {file_path}")
        else:
            await self.agent.agent_io.agent_send("\nNo chat files currently loaded.")


class ResetCommand(Command):
    command = "reset"
    description = "Reset chat history and chat files - /reset"

    async def execute(self, name: str, arg: str):
        self.agent.state.reset()
        await self.agent.agent_io.agent_send("Reset complete.")


class CommitCommand(Command):
    command = "commit"
    description = "Auto-commit changes using GitCommitAgent - /commit"

    async def execute(self, name: str, arg: str):
        commit_agent = self.agent.context.get("commit_agent")
        if not commit_agent:
            await self.agent.agent_io.agent_send("No commit agent, ignoring.")
        result = await commit_agent.auto_commit_changes()
        await self.agent.agent_io.agent_send(result)
