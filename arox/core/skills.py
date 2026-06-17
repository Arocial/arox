import logging
from pathlib import Path

from arox.utils.markdown import parse_yaml_frontmatter

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
                metadata, body = parse_yaml_frontmatter(content)

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
