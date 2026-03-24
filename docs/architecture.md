# Architecture

Arox is designed with a modular architecture that separates concerns into distinct components. This allows for high flexibility and extensibility when building AI agents.

## Core Components

### 1. Agent Patterns

Agent Patterns define the core behavior and interaction model of an AI agent.

- **`LLMBaseAgent`**: The foundational class for all LLM-driven agents. It handles:
    - Model inference and provider abstraction (via `pydantic_ai`).
    - State and history management.
    - Tool integration (both local Python functions and MCP servers).
    - Execution hooks (Pre-step and Post-step).
- **`ChatAgent`**: Extends `LLMBaseAgent` to implement a continuous conversational loop. It introduces the concept of a `CommandManager` to handle user-triggered commands during the chat.

### 2. Composers (Apps)

A **Composer** is responsible for assembling a complete application by wiring together multiple agents, tools, and an IO adapter. 

- It defines a **Main Agent** (usually a `ChatAgent` that interacts with the user) and optional **Subagents** (specialized agents that run in the background or are invoked by the main agent).
- It manages the lifecycle of all components and sets up the communication channels between them.
- Example: The `Coder` app uses a Composer to set up a main coding assistant agent alongside subagents like a `GitCommitAgent` and a `CompactionAgent`.

### 3. Plugins

Plugins are the primary way to extend an agent's capabilities. They bundle together tools, commands, and history processors into a cohesive unit.

- **Tools**: Functions provided to the LLM to interact with the external environment. Arox supports two types of tools:
    - **Local Tools**: Python functions registered directly with the agent via plugins (e.g., `Shell` execution, `ask_human`).
    - **MCP (Model Context Protocol) Tools**: Arox natively supports connecting to MCP servers via `fastmcp`, allowing agents to leverage a wide ecosystem of external tools and data sources seamlessly.
- **Commands**: Structured actions triggered by human users (e.g., `/commit`, `/reset`, `/model`). They are handled by the `CommandManager` in a `ChatAgent` and can execute local Python code or trigger specific agent behaviors without sending a prompt to the LLM, saving time and tokens.
- **History Processors**: Functions that can modify the message history before it is sent to the LLM.

### 4. Capabilities

Capabilities provide a typed, decoupled way for plugins and agents to declare what they provide or require. 

- A `Capability` is a typed object representing a specific feature or service (e.g., `FileEditCapability`).
- Plugins can provide implementations for specific capabilities.
- Other components can request a capability from the agent, allowing them to use the feature without knowing which specific plugin provides it. This promotes loose coupling and modularity.

### 5. IO Adapters (UI)

IO Adapters abstract the user interface, allowing the same agent logic to run across different platforms.

- **`TextIOAdapter`**: A rich terminal interface using `prompt-toolkit`.
- **`VercelStreamIOAdapter`**: For integration with web frontends using the Vercel AI SDK.
- **`TelegramIOAdapter`**: For running the agent as a Telegram bot.
- **`FeishuIOAdapter`**: For running the agent as a Feishu (Lark) bot.

## Data Flow

1. **User Input**: The user sends a message via the UI (handled by the IO Adapter).
2. **Command Check**: The `ChatAgent` checks if the input is a command. If so, it executes the command locally.
3. **LLM Inference**: If it's a normal message, it's appended to the state and sent to the LLM via `pydantic_ai`.
4. **Tool Execution**: If the LLM decides to call a tool, the framework executes the tool (local or MCP) and returns the result to the LLM.
5. **Response**: The final text response from the LLM is streamed back to the user via the IO Adapter.
