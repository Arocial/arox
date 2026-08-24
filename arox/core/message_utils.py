import dataclasses
from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextContent,
    UserContent,
    UserPromptPart,
)

AROX_INTERNAL_KEY = "arox_internal"


def internal_user_prompt_part(
    content: str | Sequence[UserContent],
) -> UserPromptPart:
    items: list[UserContent] = [content] if isinstance(content, str) else list(content)
    marked_items: list[UserContent] = []
    has_marker = False
    for item in items:
        if isinstance(item, str):
            marked_items.append(
                TextContent(content=item, metadata={AROX_INTERNAL_KEY: True})
            )
            has_marker = True
        elif isinstance(item, TextContent):
            metadata = dict(item.metadata) if isinstance(item.metadata, dict) else {}
            metadata[AROX_INTERNAL_KEY] = True
            marked_items.append(dataclasses.replace(item, metadata=metadata))
            has_marker = True
        else:
            marked_items.append(item)

    if not has_marker:
        marked_items.insert(
            0, TextContent(content="", metadata={AROX_INTERNAL_KEY: True})
        )
    return UserPromptPart(content=marked_items)


def _is_internal_content(content: UserContent) -> bool:
    return (
        isinstance(content, TextContent)
        and isinstance(content.metadata, dict)
        and bool(content.metadata.get(AROX_INTERNAL_KEY))
    )


def visible_message_history(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    visible_messages: list[ModelMessage] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            visible_messages.append(message)
            continue

        parts = []
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                parts.append(part)
                continue

            if any(_is_internal_content(item) for item in part.content):
                continue
            parts.append(part)

        if parts:
            visible_messages.append(dataclasses.replace(message, parts=parts))
    return visible_messages
