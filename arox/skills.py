import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def discover_skills(workspace: Path):
    """Discover skills from project and user directories."""
    scopes = [
        workspace / ".arox" / "skills",
        workspace / ".agents" / "skills",
        Path.home() / ".arox" / "skills",
        Path.home() / ".agents" / "skills",
    ]

    skills = {}
    for scope in scopes:
        if not scope.exists() or not scope.is_dir():
            continue

        for skill_dir in scope.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists() or not skill_file.is_file():
                continue

            try:
                content = skill_file.read_text(encoding="utf-8")
                if not content.startswith("---"):
                    continue

                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue

                frontmatter = parts[1]
                try:
                    metadata = yaml.safe_load(frontmatter)
                except yaml.YAMLError:
                    # Try to fix malformed YAML (e.g. unquoted colons)
                    fixed_lines = []
                    for line in frontmatter.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            v = v.strip()
                            if (
                                ":" in v
                                and not (v.startswith("'") and v.endswith("'"))
                                and not (v.startswith('"') and v.endswith('"'))
                            ):
                                fixed_lines.append(f"{k}: '{v}'")
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    fixed_frontmatter = "\n".join(fixed_lines)
                    try:
                        metadata = yaml.safe_load(fixed_frontmatter)
                    except yaml.YAMLError:
                        logger.warning(
                            f"Failed to parse YAML frontmatter in {skill_file}"
                        )
                        continue

                if (
                    not isinstance(metadata, dict)
                    or "name" not in metadata
                    or "description" not in metadata
                ):
                    logger.warning(f"Missing required metadata in {skill_file}")
                    continue

                name = metadata["name"]
                if name not in skills:
                    skills[name] = {
                        "name": name,
                        "description": metadata["description"],
                        "location": str(skill_file.absolute()),
                    }
            except Exception as e:
                logger.warning(f"Error reading skill file {skill_file}: {e}")

    return skills


def build_skill_catalog(skills: dict) -> str:
    """Build the skill catalog XML string."""
    if not skills:
        return ""

    catalog = ["<available_skills>"]
    for skill in skills.values():
        catalog.append("  <skill>")
        catalog.append(f"    <name>{skill['name']}</name>")
        catalog.append(f"    <description>{skill['description']}</description>")
        catalog.append(f"    <location>{skill['location']}</location>")
        catalog.append("  </skill>")
    catalog.append("</available_skills>")

    instructions = """
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, use your file-read tool to load
the SKILL.md at the listed location before proceeding.
When a skill references relative paths, resolve them against the skill's
directory (the parent of SKILL.md) and use absolute paths in tool calls.
"""
    return instructions + "\n" + "\n".join(catalog)
