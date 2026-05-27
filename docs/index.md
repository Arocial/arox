# Arox

**Flexible LLM-based Agents Framework**

Arox is a Python framework designed to build AI agents that improve work efficiency, particularly for software engineering tasks. It provides a structured way to define, compose, and interact with LLM-based agents.

## Goals

The primary goal of Arox is to build AI agents that act as capable assistants, automating and streamlining complex workflows. By providing a flexible architecture, Arox allows developers to create specialized agents tailored to their specific needs.

## Getting Started

> **⚠️ Security Warning**: Arox agents can execute shell commands and modify files directly on your host machine without a sandbox. They run with the same permissions as the user executing the app. Please use caution, especially when running agents on untrusted codebases or with highly capable models.

To learn more about how Arox works and how to build your own agents, explore the following sections:

- **[Architecture](architecture.md)** — runtime hierarchy (App → MainAgent), IO system, agent types, and extension points.
- **[Apps: Coder](apps/coder.md)** — the built-in Coder application.
- **[Development](development.md)** — project setup and contribution guide.
- **[Philosophy](philosophy.md)** — design philosophy behind Arox.
- **[Vercel AI API](vercel_ai_api.md)** — Vercel AI SDK compatible API reference.
