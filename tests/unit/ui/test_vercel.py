from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

from arox.core.types import USER_INPUT_ID_KEY
from arox.ui.vercel_ai import build_state_history


def test_build_state_history_carries_user_input_id():
    """The history builder must thread user_input_id onto the message metadata."""
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(content="hi\n", metadata={USER_INPUT_ID_KEY: "abc123"})
                ]
            )
        ]
    )

    history = build_state_history([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "abc123"


def test_build_state_history_untagged_user_message_stays_clean():
    """An untagged user turn must not gain an anchor."""
    request = ModelRequest(
        parts=[UserPromptPart(content=[TextContent(content="hi\n")])]
    )

    history = build_state_history([request])
    assert len(history) == 1
    msg = history[0]
    assert msg["role"] == "user"
    assert msg.get("metadata") is None or USER_INPUT_ID_KEY not in msg.get(
        "metadata", {}
    ).get("custom", {})


def test_build_state_history_identical_text_different_anchors():
    """Identical text in different parts or messages must keep their unique anchors."""
    request1 = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content="same text\n", metadata={USER_INPUT_ID_KEY: "a"}
                    )
                ]
            )
        ]
    )
    request2 = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content="same text\n", metadata={USER_INPUT_ID_KEY: "b"}
                    )
                ]
            )
        ]
    )

    history = build_state_history([request1, request2])
    assert len(history) == 2
    assert (
        history[0].get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "a"
    )
    assert (
        history[1].get("metadata", {}).get("custom", {}).get(USER_INPUT_ID_KEY) == "b"
    )
