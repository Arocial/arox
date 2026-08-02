# External Plugin Development

Arox plugins can add model tools, slash commands, Pydantic AI capabilities, history processors, lifecycle hooks, and slot integrations without changing the Arox package itself.

External plugins are standard Python packages registered through the `arox.plugins` entry-point group. Packaging provides one mechanism for plugin discovery, dependency resolution, version constraints, editable development, lockfiles, and distribution.

Installing a plugin does not enable it automatically. Its entry-point name must also appear in the target agent's `plugins` configuration.

## Create a Plugin Package

A typical package uses a `src` layout:

```text
arox-plugin-greeting/
├── pyproject.toml
├── README.md
├── src/
│   └── arox_plugin_greeting/
│       ├── __init__.py
│       └── plugin.py
└── tests/
    └── test_plugin.py
```

### Package metadata and dependencies

Declare the plugin's dependencies in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "arox-plugin-greeting"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "arox",
    "httpx>=0.28,<1",
]

[project.entry-points."arox.plugins"]
greeting = "arox_plugin_greeting.plugin:GreetingPlugin"

[tool.setuptools.packages.find]
where = ["src"]
```

The entry point has two parts:

```text
greeting = "arox_plugin_greeting.plugin:GreetingPlugin"
^ name      ^ module                        ^ class
```

- `greeting` is the name used in Arox configuration.
- `arox_plugin_greeting.plugin` is an importable module in the package.
- `GreetingPlugin` is the exported plugin class.

Keep dependency declarations in `project.dependencies`. Do not duplicate them in Arox configuration or a separate plugin manifest.

## Implement the Plugin

Every plugin class must extend `arox.core.plugin.Plugin`:

```python
# src/arox_plugin_greeting/plugin.py
from arox.core.plugin import Plugin, tool


class GreetingPlugin(Plugin):
    @tool()
    async def greet(self, name: str) -> str:
        """Greet a person by name."""
        prefix = self.config.get("prefix", "Hello")
        return f"{prefix}, {name}!"
```

The inherited constructor receives the owning agent. Arox applies the plugin's per-agent configuration before collecting its capabilities.

Common extension points are:

- `@tool()` methods exposed to the model.
- `commands()` for slash-command handlers.
- `capabilities()` for additional Pydantic AI capabilities.
- `history_processor()` for modifying model history before a request.
- `on_load()` for one-time agent wiring, including slot providers.
- `on_start()` and `on_stop()` for async resource lifecycle management.

## Install the Plugin

The plugin must be installed in the same Python environment that runs Arox. Installing it in an unrelated virtual environment does not make its entry point visible to Arox.

### Local editable development

Install a local package in editable mode while developing it:

```bash
uv pip install -e /path/to/arox-plugin-greeting
```

If Arox is run from a uv-managed application project and the plugin should be recorded in its lockfile, add it as an editable path dependency instead:

```bash
uv add --editable --no-workspace /path/to/arox-plugin-greeting
```

Editable installation means Python source changes are used without rebuilding the package. Restart Arox or construct a new agent after changing plugin code because active agents retain their existing plugin instances and capabilities.

### Install a released package

Install a released plugin alongside Arox:

```bash
uv add arox-plugin-greeting
```

For an isolated `uv tool` installation, include the plugin in the Arox tool environment rather than installing it as a separate tool:

```bash
uv tool install --with arox-plugin-greeting arox
```

The package installer resolves Arox and all enabled plugin requirements together. If two plugins require incompatible versions of the same dependency, installation should fail rather than deferring the conflict until runtime.

## Enable and Configure the Plugin

Use the entry-point name in the agent's plugin list:

```toml
[agent.coder]
plugins = [
    "core",
    "greeting",
]

[agent.coder.plugin_config.greeting]
prefix = "Welcome"
```

The `plugin_config` key must match the entry-point name used in `plugins`. The plugin receives a copy of that mapping through `self.config`.

Plugins are instantiated in `AgentConfig.plugins` order. This order also determines command and slot registration order and the order in which `on_start()` is called.

## Tools

Use `@tool()` to expose plugin methods to the model:

```python
from arox.core.plugin import Plugin, tool


class StoragePlugin(Plugin):
    @tool(sequential=True)
    async def write_record(self, key: str, value: str) -> str:
        """Write a record to external storage."""
        await self.client.write(key, value)
        return "Record written"
```

Set `sequential=True` when a tool must not run concurrently with other sequential tools.

Tool exposure can depend on plugin configuration:

```python
class StoragePlugin(Plugin):
    @tool(enabled=lambda plugin: plugin.config.get("allow_delete", False))
    async def delete_record(self, key: str) -> str:
        """Delete a record from external storage."""
        await self.client.delete(key)
        return "Record deleted"
```

The predicate is evaluated after Arox applies the plugin configuration.

## Lifecycle Management

Use `on_start()` and `on_stop()` for resources such as HTTP clients, database connections, and subprocesses:

```python
import httpx

from arox.core.plugin import Plugin, tool


class ApiPlugin(Plugin):
    client: httpx.AsyncClient | None = None

    async def on_start(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.config["base_url"],
            timeout=30,
        )

    async def on_stop(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    @tool()
    async def fetch_item(self, item_id: str) -> str:
        """Fetch an item from the configured API."""
        if self.client is None:
            raise RuntimeError("Plugin has not started")
        response = await self.client.get(f"/items/{item_id}")
        response.raise_for_status()
        return response.text
```

Arox starts plugins in configuration order and closes their resources in reverse order when the agent stops. Constructors and `on_load()` should not assume that async resources are already active.

## Test the Plugin

Plugin logic can be tested directly with a minimal agent stub:

```python
from types import SimpleNamespace

import pytest

from arox_plugin_greeting.plugin import GreetingPlugin


@pytest.mark.asyncio
async def test_greeting_uses_configured_prefix():
    plugin = GreetingPlugin(SimpleNamespace())
    plugin.configure({"prefix": "Welcome"})

    assert await plugin.greet("Arox") == "Welcome, Arox!"
```

An integration test can verify that package metadata exposes the expected entry point:

```python
from importlib.metadata import entry_points


def test_plugin_entry_point():
    matches = [
        entry_point
        for entry_point in entry_points(group="arox.plugins")
        if entry_point.name == "greeting"
    ]

    assert len(matches) == 1
    assert matches[0].load().__name__ == "GreetingPlugin"
```

For model-level integration tests, construct an Arox agent configured with the plugin and use Pydantic AI's deterministic test models to inspect the exposed tools.

## Security and Isolation

Plugins execute in the Arox process with the same permissions as Arox. They are not sandboxed and can access files, environment variables, credentials, and network resources available to the process.

All plugins also share one Python environment. Standard packaging can resolve compatible dependency constraints, but it cannot load mutually incompatible versions of the same library into one process. A plugin that requires strict dependency or security isolation should run as a separate service or subprocess and communicate with Arox through an explicit protocol such as MCP.

Only install plugins from trusted sources, and review their dependencies, build configuration, lifecycle hooks, and tool implementations.

## Troubleshooting

### Plugin cannot be resolved

Check that:

- The package is installed in the same environment that executes Arox.
- `pyproject.toml` registers the expected name under `[project.entry-points."arox.plugins"]`.
- The entry-point name exactly matches the value in `AgentConfig.plugins`.
- Arox was restarted after the plugin was installed.

You can inspect the entry points visible to the current interpreter:

```bash
python - <<'PY'
from importlib.metadata import entry_points

for entry_point in entry_points(group="arox.plugins"):
    print(f"{entry_point.name}: {entry_point.value}")
PY
```

### Plugin import fails

Install or synchronize the plugin package rather than installing individual imports manually. Its `project.dependencies` should describe everything needed at runtime.

### Configuration is missing

Ensure that the `plugin_config` table uses the same key as the plugin entry point:

```toml
[agent.coder.plugin_config.greeting]
prefix = "Welcome"
```

### Plugin edits are not visible

Confirm that the package was installed in editable mode, then restart Arox or construct a new agent. Existing agents do not rebuild their plugin capabilities during configuration reload.
