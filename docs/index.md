# Arox

**Flexible LLM-based Agents Framework**

Arox is a Python framework designed to build AI agents that improve work efficiency, particularly for software engineering tasks. It provides a structured way to define, compose, and interact with LLM-based agents.

## Goals

The primary goal of Arox is to build AI agents that act as capable assistants, automating and streamlining complex workflows. By providing a flexible architecture, Arox allows developers to create specialized agents tailored to their specific needs.

## Core Concepts

Arox is built around several key concepts that work together to create powerful AI applications:

### Agent Patterns

Agent Patterns are reusable templates for creating AI agents with specific behaviors and capabilities. They provide a structured way to define how agents should interact with users, tools, and other agents. 

Examples include:

- `ChatAgent`: For conversational interactions.
- `LLMBaseAgent`: The foundation for all LLM-driven agent types.

### Composed Agents (Apps)

Composed Agents (often referred to as Apps) are specialized agents built by combining multiple Agent Patterns and tools. They are designed to handle complex workflows, such as code generation or repository management. 

Example:

- `CoderComposer`: An application designed to assist with software development tasks.

### Plugins
Plugins are modular components that extend the capabilities of an agent. They bundle together:

- **Tools**: External functions provided to the LLM to interact with the outside world (e.g., reading files, executing shell commands).
- **Commands**: Predefined actions triggered by user input (e.g., `/commit`, `/reset`), providing a structured way for humans to interact with agents.
- **History Processors**: Functions that can modify the message history before it is sent to the LLM.

### Capabilities

Capabilities provide a typed, decoupled way for plugins and agents to declare what they provide or require. This allows different components to interact without tight coupling. For example, a plugin might provide a `FileEditCapability`, which another component can consume to modify files.

## Getting Started

To learn more about how Arox works and how to build your own agents, explore the following sections:

- **[Architecture](architecture.md)**: Deep dive into the core components of the framework.
- **[Apps: Coder](apps/coder.md)**: Learn about the built-in Coder application.
- **[Development](development.md)**: Guide on how to set up the project and contribute.
- **[Philosophy](philosophy.md)**: Read about the design philosophy behind Arox.
