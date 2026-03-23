import json
import logging
import uuid
from typing import Any

import yaml
from pydantic_ai import RunContext
from pydantic_ai.exceptions import CallDeferred

from arox.agent_patterns.llm_base import AgentDeps
from arox.agent_patterns.plugin import Command, Plugin

logger = logging.getLogger(__name__)


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


class CompactionCommand(Command):
    command = "compact"
    description = "Compact conversation history - /compact"

    async def execute(self, name: str, arg: str):
        compaction_agent = self.agent.context.get("compaction_agent")
        if not compaction_agent:
            await self.agent.agent_io.agent_send("No compaction agent configured.")
            return

        example_len = len(self.agent.state.example_messages)
        messages_to_compact = self.agent.state.message_history[example_len:]

        if not messages_to_compact:
            await self.agent.agent_io.agent_send("No history to compact.")
            return

        summary = await compaction_agent.compact(messages_to_compact)

        from pydantic_ai import ModelRequest, UserPromptPart

        new_request = ModelRequest(
            parts=[UserPromptPart(content=f"Previous conversation summary:\n{summary}")]
        )

        self.agent.state.message_history = self.agent.state.example_messages + [
            new_request
        ]

        await self.agent.agent_io.agent_send(
            "Conversation history compacted successfully."
        )


async def ask_human(ctx: RunContext["AgentDeps"], question: str) -> str:
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
    await ctx.deps.agent_io.add_tool_input_request(question, key)

    async def callback():
        return await ctx.deps.agent_io.get_tool_input_result(key)

    raise CallDeferred(metadata={"result_callback": callback})


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
        tools: list[dict[str, Any]] = [{"func": ask_human}]
        return tools
