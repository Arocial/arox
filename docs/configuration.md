# Configuration

Arox features a flexible, hierarchical configuration system. Configurations are loaded from various sources and merged using a "**deep merge**" strategy. When the same key is defined in multiple sources, the value from the source with the higher precedence overrides the lower one.

## Resolution Precedence

Configuration sources are evaluated and merged in an interleaved, scope-based order. **Later scopes override earlier scopes.**

The loading order (from lowest to highest precedence) is as follows:

1. **Default Values**: Hardcoded in the Pydantic models.
2. **Global Scope**:
    - **Global Agent & Skill Files**: Auto-discovered `.md` agent definitions and `SKILL.md` skill files from:
        - `~/.config/arox/agents` / `~/.config/arox/skills` (or equivalent `user_config_dir` for your platform)
        - `~/.agents/agents` / `~/.agents/skills`
    - **Global Config File**: Parsed from `~/.config/arox/config.[toml|yaml]`.
    *(The global config file will override any conflicting settings defined in the global agent files.)*
3. **App Profile Scope** *(if applicable)*:
    - **App Agent & Skill Files**: Auto-discovered from profile directories like `~/.config/arox/profiles/{app_name}/{profile}/agents` and `skills/`.
    - **App Config File**: Parsed from profile directory like `~/.config/arox/profiles/{app_name}/{profile}/config.[toml|yaml]`.
4. **Workspace Scope**:
    - **Workspace Agent & Skill Files**: Auto-discovered `.md` agent definitions and `SKILL.md` skill files from:
        - `$WORKSPACE/.arox/agents` / `$WORKSPACE/.arox/skills`
        - `$WORKSPACE/.agents/agents` / `$WORKSPACE/.agents/skills`
    *(Workspace-level agents will gracefully override Global/Profile Agents and the Global Config.)*
    - **Workspace Config File**: Parsed from `$WORKSPACE/.arox/config.[toml|yaml]`.
    *(The workspace config file will override anything defined previously, including workspace agent files.)*
5. **CLI Overrides**: Explicit dot-notation key-value overrides passed via command line (e.g., `--app.main_agent=custom`). These have the absolute highest precedence.

### Agent File Discovery (`.md` frontmatter)

Arox automatically discovers agent definitions within `agents/` directories across various configuration scopes.

These files should be Markdown files. The system parses the YAML frontmatter blocks (`---` ... `---`) to extract configuration keys. If a `system_prompt` is not explicitly defined in the frontmatter, the rest of the Markdown body is automatically treated as the `system_prompt`.

Example `.arox/agents/my_agent.md`:
```markdown
---
type: "chat"
model_ref: "deepseek:deepseek-chat"
---
You are a helpful assistant.
```

### File Formats and `include` Directive

Main configuration files (like `config.toml` or `config.yaml`) support `toml`, `yaml`, and `yml` formats. 

These files also support a top-level `include` directive, which can be a string or a list of strings pointing to other configuration files (paths are resolved relative to the host file). 
* Included files are merged sequentially.
* The content of the host file overrides the included files.
* Circular includes will throw a `ValueError`.

Example `.arox/config.toml`:
```toml
include = ["shared-settings.toml"]

[app]
main_agent = "coder"
```

## Environment Variables

Arox utilizes the `platformdirs` library to determine conventional paths for your operating system. For example, on Linux it respects the `XDG_CONFIG_HOME` specification and will use `~/.config/arox/`.
