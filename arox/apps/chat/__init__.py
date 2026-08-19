from typing import Any


def main(profile: str | None = None) -> Any:
    """Run the chat app without eagerly importing its composition root."""
    from arox.apps.chat.main import main as run

    return run(profile)


__all__ = ["main"]
