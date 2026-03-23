# Development Guide

This guide provides instructions on how to set up the development environment, run tests, and contribute to the Arox framework.

## Prerequisites

- **Python**: Version `>=3.12` is required.
- **Package Manager**: We use [`uv`](https://github.com/astral-sh/uv) for fast and reliable package management.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Arocial/arox.git
   cd arox
   ```

2. **Install dependencies**:
   Use `uv` to install the project and its development dependencies:
   ```bash
   uv sync
   ```

## Project Structure

- `arox/`: The main framework code.
    - `agent_patterns/`: Core agent logic (`LLMBaseAgent`, `ChatAgent`, `Composer`).
    - `apps/`: Built-in applications (e.g., `coder`).
    - `codebase/`: Utilities for interacting with codebases (e.g., file editing).
    - `commands/`: Built-in commands for agents.
    - `tools/`: Built-in tools (e.g., `shell`).
    - `ui/`: IO Adapters for different platforms (Text, Vercel AI, Telegram, Feishu).
    - `utils/`: Helper functions.
- `tests/`: Unit and functional tests.
- `tools/`: Development scripts (e.g., `lint`).
- `docs/`: Documentation files.

## Testing

We use `pytest` for testing. To run the test suite:

```bash
uv run pytest
```

## Linting and Formatting

We use `ruff` for fast linting and formatting. A convenience script is provided in the `tools/` directory.

To run the linter and formatter:

```bash
./tools/lint
```

## Building Documentation

The documentation is built using MkDocs with the Material theme.

To serve the documentation locally (it will be available at `http://127.0.0.1:3420`):

```bash
uv run mkdocs serve
```

To build the static site:

```bash
uv run mkdocs build
```
