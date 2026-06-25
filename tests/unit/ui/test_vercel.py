import pytest
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart

import arox.ui.vercel_ai  # noqa: F401 -- import applies the dump_messages patch
from arox.core.types import USER_INPUT_ID_KEY


def test_dump_messages_carries_user_input_id():
    """The monkey patch must thread user_input_id onto the dumped TextUIPart.

    Guards against silent breakage if pydantic_ai renames or reshapes
    ``_convert_user_prompt_part``: without the patch the anchor is dropped and
    the assertion fails.
    """
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(content="hi\n", metadata={USER_INPUT_ID_KEY: "abc123"})
                ]
            )
        ]
    )

    [ui_message] = VercelAIAdapter.dump_messages([request])
    [part] = ui_message.parts
    assert isinstance(part, TextUIPart)
    assert part.provider_metadata == {"arox": {USER_INPUT_ID_KEY: "abc123"}}


def test_dump_messages_untagged_user_message_stays_clean():
    """An untagged user turn must not gain a provider_metadata wrapper."""
    request = ModelRequest(
        parts=[UserPromptPart(content=[TextContent(content="hi\n")])]
    )

    [ui_message] = VercelAIAdapter.dump_messages([request])
    [part] = ui_message.parts
    assert isinstance(part, TextUIPart)
    assert part.provider_metadata is None


def test_dump_messages_rejects_ambiguous_anchors_in_one_part():
    """Two same-text TextContents with different anchors must assert, not guess.

    Text-keyed matching only works while text -> id is unambiguous within a
    part; this documents and enforces that precondition.
    """
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(content="hi\n", metadata={USER_INPUT_ID_KEY: "a"}),
                    TextContent(content="hi\n", metadata={USER_INPUT_ID_KEY: "b"}),
                ]
            )
        ]
    )

    with pytest.raises(AssertionError):
        VercelAIAdapter.dump_messages([request])
