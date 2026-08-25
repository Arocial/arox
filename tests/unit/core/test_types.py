import uuid

import pytest

from arox.core.types import ClientInput, MessagePayload, normalize_client_input


def _message_input(content, *, client_message_id=None):
    return normalize_client_input(
        ClientInput(
            payload=MessagePayload(content=content),
            client_message_id=client_message_id,
        )
    )


@pytest.mark.parametrize("client_message_id", [None, ""])
def test_user_input_generates_missing_client_message_id(client_message_id):
    user_input = _message_input("hello", client_message_id=client_message_id)

    assert user_input.client_message_id is not None
    uuid.UUID(user_input.client_message_id)


def test_user_input_preserves_client_message_id():
    user_input = _message_input("hello", client_message_id="client-1")

    assert user_input.client_message_id == "client-1"
