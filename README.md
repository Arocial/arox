# Arox

**Flexible LLM-based Agents Framework**

Arox is a Python framework designed to build AI agents that improve work efficiency, particularly for software engineering tasks. It provides a structured way to define, compose, and interact with LLM-based agents.

## Goals

The primary goal of Arox is to build AI agents that act as capable assistants, automating and streamlining complex workflows. By providing a flexible architecture, Arox allows developers to create specialized agents tailored to their specific needs.

## Core Concepts

Arox is built around several key concepts that work together to create powerful AI applications:

### Agent Patterns
Agent Patterns are reusable templates for creating AI agents with specific behaviors and capabilities. They provide a structured way to define how agents should interact with users, tools, and other agents. Examples include `ChatAgent` for conversational interactions and `LLMBaseAgent` as the foundation for all agent types.

### Composed Agents (Apps)
Composed Agents (often referred to as Apps) are specialized applications built by combining multiple Agent Patterns and tools via a `Composer`. They are designed to handle complex workflows, such as code generation or repository management. The primary example is the `Coder` app.

### Plugins
Plugins are modular components that extend the capabilities of an agent. They bundle together:

- **Tools**: External functions provided to the LLM to interact with the outside world (e.g., reading files, executing shell commands). Arox supports both local Python functions and MCP (Model Context Protocol) servers.
- **Commands**: Predefined actions triggered by user input (e.g., `/commit`, `/reset`), providing a structured way for humans to interact with agents.
- **History Processors**: Functions that can modify the message history before it is sent to the LLM.

### Capabilities
Capabilities provide a typed, decoupled way for plugins and agents to declare what they provide or require. This allows different components to interact without tight coupling. For example, a plugin might provide a `FileEditCapability`, which another component can consume to modify files.

### UI Adapters
Arox supports multiple user interfaces through IO Adapters, allowing the same agent logic to run in a terminal (`TextIOAdapter`), a web frontend (`VercelStreamIOAdapter`), or as a chat bot (`TelegramIOAdapter`, `FeishuIOAdapter`).

## Documentation

For more detailed information, please refer to the documentation:

- [Architecture](docs/architecture.md)
- [Apps: Coder](docs/apps/coder.md)
- [Development Guide](docs/development.md)
- [Design Philosophy](docs/philosophy.md)

## Quick Start

1. Install the package using `uv`:
   ```bash
   uv sync
   ```
2. Run the Coder app:
   ```bash
   uv run arox-coder
   ```
