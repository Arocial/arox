from typing import TypeVar

T = TypeVar("T")


class Capability[T]:
    """
    Represents a typed capability that can be provided and consumed.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Capability):
            return self.name == other.name
        return False
