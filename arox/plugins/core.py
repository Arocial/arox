import json
import logging
import uuid

import yaml
from pydantic_ai.exceptions import CallDeferred

from arox.agent_patterns.plugin import Plugin, command, tool
from arox.plugins.capabilities import AGENT_INFO, AGENT_RESET, SUBAGENT

logger = logging.getLogger(__name__)


class CorePlugin(Plugin):
    @command("model", "Switch LLM model - /model <model_name>")
    async def model_command(self, name: str, arg: str):
        if not arg:
            await self.agent.agent_io.agent_send("Please specify a model name")
            return
        self.agent.set_model(arg)

    @command(
        "invoke-tool",
        "Invoke a registered tool - /invoke-tool <function_name> [json_args]",
    )
    async def invoke_tool_command(self, name: str, arg: str):
        tool_registry = self.agent.tool_registry

        parts = arg.split(maxsplit=1)
        if len(parts) < 1:
            await self.agent.agent_io.agent_send(
                "Usage: /invoke-tool <function_name> [json_args]"
            )
            return

        function_name = parts[0]
        args_str = parts[1] if len(parts) > 1 else "{}"

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

    @command("list-tools", "List all registered tools")
    async def list_tools_command(self, name: str, arg: str):
        tool_registry = self.agent.tool_registry
        tool_specs = await tool_registry.get_tools_specs()
        if not tool_specs:
            await self.agent.agent_io.agent_send("No tools registered.")
            return

        await self.agent.agent_io.agent_send("Registered Tools:")
        await self.agent.agent_io.agent_send(yaml.safe_dump(tool_specs))

    @command("info", "Show current chat files and model in use - /info")
    async def info_command(self, name: str, arg: str):
        # Show current model
        current_model = getattr(self.agent, "provider_model", "Unknown")
        await self.agent.agent_io.agent_send(f"Current model: {current_model}")

        for provider in self.agent.get_capability(AGENT_INFO):
            info = await provider()
            if info:
                await self.agent.agent_io.agent_send(info)

    @command("reset", "Reset chat history and chat files - /reset")
    async def reset_command(self, name: str, arg: str):
        self.agent.reset()
        for provider in self.agent.get_capability(AGENT_RESET):
            provider()
        await self.agent.agent_io.agent_send("Reset complete.")

    @command("agent", "Call a subagent - /agent <name> [task]")
    async def agent_command(self, name: str, arg: str):
        parts = arg.split(maxsplit=1)
        if not parts:
            await self.agent.agent_io.agent_send("Usage: /agent <name> [task]")
            return

        subagent_name = parts[0]
        task = parts[1] if len(parts) > 1 else ""

        subagent = None
        for get_subagent_func in self.agent.get_capability(SUBAGENT):
            subagent = get_subagent_func(subagent_name)
            if subagent:
                break

        if not subagent:
            await self.agent.agent_io.agent_send(
                f"Subagent '{subagent_name}' not found."
            )
            return

        if hasattr(subagent, "handle_task"):
            result = await subagent.handle_task(task, main_agent=self.agent)
            if result:
                await self.agent.agent_io.agent_send(result)
        else:
            await self.agent.agent_io.agent_send(
                f"Subagent '{subagent_name}' does not support tasks."
            )

    @tool()
    async def ask_human(self, question: str) -> str:
        """
        Ask human for more information or decisions.

        Use this tool when the current task requires more input or information from the user.
        Scenarios include, but are not limited to:
        - Multiple viable options are available and user decision is required.
        - Critical information is missing and needs to be provided by the user.
        - Confirming destructive or high-risk operations (e.g., deleting databases, overwriting critical files).
        - Clarifying ambiguous requirements or instructions.
        - Requesting credentials, API keys, or sensitive data that should not be guessed.
        """
        key = str(uuid.uuid4())
        await self.agent.agent_io.add_tool_input_request(question, key)

        async def callback():
            return await self.agent.agent_io.get_tool_input_result(key)

        raise CallDeferred(metadata={"result_callback": callback})
