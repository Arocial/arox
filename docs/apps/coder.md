# Coder App

The **Coder** app is the primary application built on top of the Arox framework. It is designed to act as an AI coding assistant specialized in software development tasks.

## Overview

The Coder app uses a `Composer` to assemble a main `ChatAgent` (the "coder") along with several subagents that handle specific background tasks, such as generating Git commit messages and compacting conversation history.

## Architecture

The Coder app consists of the following components:

- **Main Agent (`coder`)**: A `ChatAgent` that interacts directly with the user. It is equipped with tools for reading and writing files, executing shell commands, and interacting with the codebase.
- **Subagent (`git_commit_agent`)**: A specialized agent responsible for automatically generating meaningful Git commit messages based on the changes made by the main agent.
- **Subagent (`compaction`)**: A specialized agent that summarizes long technical conversations to manage the context window size and improve LLM performance.

## Features

- **Code Editing**: The agent can read, modify, and create files in your project. It is instructed to only edit files when it knows the exact current content of the section being modified.
- **Shell Execution**: The agent can run shell commands (e.g., `ls`, `grep`, `pytest`) to explore the codebase and verify its changes.
- **Automatic Commits**: After completing a feature or solving an issue, the agent can automatically commit its changes with a generated commit message, including a `Co-authored-by` tag.
- **Efficient Tool Usage**: The agent is instructed to perform multiple tool calls concurrently whenever the logic allows, avoiding unnecessary sequential interactions.
- **Commands**: The Coder app supports several built-in commands for quick actions:
  - `/project`: Manage project settings.
  - `/model`: Switch the underlying LLM model.
  - `/invoke_tool`: Manually invoke a specific tool.
  - `/list_tools`: List all available tools.
  - `/reset`: Reset the conversation history.
  - `/info`: Display information about the current agent and model.
  - `/commit`: Trigger the Git commit agent manually.
  - `/compaction`: Trigger the compaction agent manually.

## Configuration

The Coder app is configured via a TOML file (`arox/apps/coder/config.toml`). You can override these settings using command-line arguments or a custom configuration file.

### System Prompt

The core behavior of the Coder agent is defined by its system prompt, which enforces rules such as:
- Using comments sparingly and focusing on "why" rather than "what".
- Never using comments to communicate with the user.
- Preserving existing comments unless invalidated by changes.

### Model Parameters

By default, the Coder agent uses a `temperature` of `0` to ensure deterministic and reliable code generation.

## Usage

To start the Coder app, run the following command:

```bash
arox-coder
```

You can specify the UI interface using the `--ui` flag:

```bash
# Use the default text-based terminal UI
arox-coder --ui text

# Use the Vercel AI SDK interface
arox-coder --ui vercel_ai

# Run as a Telegram bot
arox-coder --ui telegram

# Run as a Feishu bot
arox-coder --ui feishu
```
