from typing import TypeVar

T = TypeVar("T")


class Slot[T]:
    """
    Represents a typed slot that can be filled by providers and read by consumers.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Slot):
            return self.name == other.name
        return False
