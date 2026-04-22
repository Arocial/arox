# Arox

**Flexible LLM-based Agents Framework**

Arox is a Python framework designed to build AI agents that improve work efficiency, particularly for software engineering tasks. It provides a structured way to define, compose, and interact with LLM-based agents.

## Goals

The primary goal of Arox is to build AI agents that act as capable assistants, automating and streamlining complex workflows. By providing a flexible architecture, Arox allows developers to create specialized agents tailored to their specific needs.

## Quick Start

> **⚠️ Security Warning**: Arox agents can execute shell commands and modify files directly on your host machine without a sandbox. They run with the same permissions as the user executing the app. Please use caution, especially when running agents on untrusted codebases or with highly capable models.

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Run the Coder app:
   ```bash
   uv run arox-coder
   ```

## Documentation

- [Architecture](docs/architecture.md) — runtime hierarchy, IO system, agent types, and extension points
- [Apps: Coder](docs/apps/coder.md)
- [Development Guide](docs/development.md)
- [Design Philosophy](docs/philosophy.md)
- [Vercel AI API](docs/vercel_ai_api.md)
