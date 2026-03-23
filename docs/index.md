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

### Tools

Tools are external components provided to the LLM to extend its capabilities. They allow agents to interact with the outside world, such as reading files, executing shell commands, or querying APIs.

### Commands

Commands are predefined actions that agents can execute, often triggered by user input. They provide a way for humans to interact with agents in a structured manner. Commands may use tools as their backend, bridging the gap between human intent and agent execution.

## Getting Started

To learn more about how Arox works and how to build your own agents, explore the following sections:

- **[Architecture](architecture.md)**: Deep dive into the core components of the framework.
- **[Apps: Coder](apps/coder.md)**: Learn about the built-in Coder application.
- **[Development](development.md)**: Guide on how to set up the project and contribute.
- **[Philosophy](philosophy.md)**: Read about the design philosophy behind Arox.
