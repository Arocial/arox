# Creating an App Profile

An Arox profile is a reusable bundle of configuration, agent definitions, and skills for one app. Profiles can reference installed external plugin packages when a workflow needs custom tools. They are useful when a workflow should be available from multiple workspaces without copying its files into every workspace.

For the `chat` app, a named profile normally lives at:

```text
~/.config/arox/profiles/chat/<profile-name>/
```

Start it with:

```bash
arox --profile <profile-name>
```

## Create the profile structure

A practical multi-agent profile can use this layout:

```text
my-profile/
├── config.toml
├── agents/
│   ├── coordinator.md
│   └── researcher.md
└── skills/
    └── source-review/
        └── SKILL.md
```

Only `config.toml` is required. Add the other directories when the profile needs those extension types.

## Configure the main agent

`config.toml` selects the user-facing agent and can configure agents discovered from `agents/`:

```toml
[app]
main_agent = "coordinator"

[agent.coordinator]
type = "chat"
plugins = ["core", "subagent"]
subagents = ["researcher"]

[agent.coordinator.plugin_config.subagent]
mode = "advanced"
```

The name in `app.main_agent` must match an entry under `agent` or the name of a discovered agent file. Likewise, every name in `subagents` must resolve to an agent configuration.

Use `mode = "advanced"` when the main agent should run independent subagent tasks concurrently or resume them later. Use the default `simple` mode when synchronous, one-shot delegation is sufficient.

## Define agents in Markdown

Agent files use YAML frontmatter for configuration and the Markdown body as the system prompt:

```markdown
---
type: chat
description: Collects and assesses evidence for the coordinator.
plugins:
  - research_data
---
You are an evidence-focused researcher. Use the available tools before making
claims, distinguish facts from inference, and report missing data explicitly.
```

The filename becomes the agent name, so `agents/researcher.md` defines `researcher`. The `description` is especially useful for subagents because Arox includes it in the delegation guidance shown to the parent agent.

Settings in `config.toml` take precedence over settings in agent-file frontmatter. Keep the agent's prompt and its local settings together in the Markdown file, and reserve `config.toml` for app wiring and shared overrides.

## Add an external plugin package

Profile-specific tools should still be distributed as standard Python packages. Register the plugin under the `arox.plugins` entry-point group:

```toml
# plugin/pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "arox-plugin-research-data"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["arox"]

[project.entry-points."arox.plugins"]
research_data = "arox_plugin_research_data:ResearchDataPlugin"
```

```python
# plugin/arox_plugin_research_data/__init__.py
from arox.core.plugin import Plugin, tool


class ResearchDataPlugin(Plugin):
    @tool()
    async def lookup_record(self, record_id: str) -> dict[str, str]:
        """Look up a research record by ID."""
        return {"id": record_id, "status": "available"}
```

Install the package in the same Python environment that runs `arox`. For local development, use an editable installation:

```bash
uv pip install -e /path/to/profile-repository/plugin
```

Installing the package does not enable it. Add its entry-point name to the `plugins` list of every agent that should receive its tools. Per-agent plugin settings can be supplied in `config.toml`:

```toml
[agent.researcher.plugin_config.research_data]
endpoint = "https://example.invalid/api"
```

Installing the plugin only in an unrelated workflow virtual environment is insufficient when `arox` is an isolated tool installation. See [External Plugin Development](external_plugins.md) for dependency management, lifecycle hooks, conditional tools, packaging, and testing details.

## Keep the profile in version control

Instead of maintaining source files directly under `~/.config`, keep the profile in its own repository and link it into Arox's profile directory:

```bash
mkdir -p ~/.config/arox/profiles/chat
ln -s /path/to/profile-repository/profile \
  ~/.config/arox/profiles/chat/my-profile
```

The repository might then look like:

```text
profile-repository/
├── profile/
│   ├── config.toml
│   └── agents/
├── plugin/
│   ├── pyproject.toml
│   └── arox_plugin_research_data/
├── tests/
└── README.md
```

This avoids hidden-directory ignore rules that commonly affect workspace `.arox/` directories and makes profile changes straightforward to review and commit. Check an existing destination before creating the link; `ln -s` should not replace a real profile directory or an unrelated link.

## Workspace and path behavior

The profile supplies reusable configuration, but the process working directory remains the agent workspace. This distinction matters for plugins that accept file paths:

- Resolve profile-owned assets relative to the plugin source location.
- Resolve user project files relative to `agent.workspace`.
- Prefer absolute or explicitly configured paths when a relative path would be ambiguous.

The profile is one layer in Arox's configuration hierarchy. Workspace agent files and `$WORKSPACE/.arox/config.toml` can override profile settings, and CLI overrides have the highest precedence. See [Configuration](configuration.md) for the complete loading order.

## Validate the profile

First confirm that the profile starts and that its agents and tools are visible:

```bash
arox --profile my-profile
```

Inside the text UI, `/info` shows the active agent and `/list_tools` shows its tools. Exercise at least one tool-backed request to verify that plugin dependencies and configured paths work at runtime.

For automated validation, load the exact app, profile, and workspace combination used in production:

```python
from importlib.metadata import entry_points
from pathlib import Path

from arox.core.config import ConfigLoader

loader = ConfigLoader(
    app_name="chat",
    profile="my-profile",
    workspace=Path.cwd(),
)
config = loader.current_config
plugin_names = {
    entry_point.name for entry_point in entry_points(group="arox.plugins")
}

assert config.app.main_agent == "coordinator"
assert "research_data" in plugin_names
```

Checking the entry point catches installation and environment mistakes before the profile starts. Restart the application after changing plugin Python code because configuration reload does not rebuild existing plugin instances.

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Profile name loads default behavior | Profile is under the wrong app directory | Put it under `profiles/chat/<name>` for the `chat` app |
| Agent is not available for delegation | Agent name is absent or mismatched | Match `subagents` entries to agent filenames or config keys |
| Plugin entry point is missing | Plugin package is installed in a different environment | Install the package in the environment that executes `arox` |
| Plugin is installed but has no tools | Plugin entry-point name was not enabled for the agent | Add the entry-point name to the agent's `plugins` list |
| Relative data path points somewhere unexpected | Profile location was confused with workspace | Resolve project data from `agent.workspace` or configure an absolute path |
| Plugin edits do not appear in a running session | Existing agents retain their plugin instances | Restart Arox or create a new agent |

