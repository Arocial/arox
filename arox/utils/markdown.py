import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    """Parse YAML frontmatter from a markdown string.

    Returns:
        A tuple of (metadata_dict, markdown_body).
        If no frontmatter is found, returns (None, content).
        If frontmatter is malformed and cannot be parsed, returns (None, content).
    """
    if not content.startswith("---"):
        return None, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, content

    frontmatter = parts[1]
    body = parts[2].strip()

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
            logger.warning("Failed to parse YAML frontmatter")
            return None, content

    if not isinstance(metadata, dict):
        return None, content

    return metadata, body
