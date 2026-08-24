import uuid

import pytest

from arox.core.types import UserInput


@pytest.mark.parametrize("client_message_id", [None, ""])
def test_user_input_generates_missing_client_message_id(client_message_id):
    user_input = UserInput(input_content="hello", client_message_id=client_message_id)

    assert user_input.client_message_id is not None
    uuid.UUID(user_input.client_message_id)


def test_user_input_preserves_client_message_id():
    user_input = UserInput(input_content="hello", client_message_id="client-1")

    assert user_input.client_message_id == "client-1"
