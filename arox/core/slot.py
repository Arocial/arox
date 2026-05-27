from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class ResultAggregator(Enum):
    """Strategy for aggregating results from multiple slot providers.

    * ``DISCARD`` – invoke every provider as a handler, discard return values
      (fire-and-forget event channel, replaces the old ``push=True``).
    * ``FIRST``  – return the first registered provider without calling it
      (single-valued extension point, replaces ``get_one``).
    * ``LIST``   – return all registered providers as a list without calling
      them (multi-valued extension point, replaces ``get_slot``).
    """

    DISCARD = "discard"
    FIRST = "first"
    LIST = "list"


class Slot[T]:
    """
    Represents a typed slot that can be filled by providers and read by consumers.

    Each slot carries a :class:`ResultAggregator` that controls how results
    are collected when the slot is emitted via :meth:`LLMBaseAgent.invoke_slot`:

    * ``DISCARD`` – the slot is an event channel: registered handlers are
      invoked and their return values are discarded.
    * ``FIRST``  – the slot is a single-valued pull extension point: only the
      first registered provider is returned.
    * ``LIST``   – the slot is a multi-valued pull extension point: all
      registered providers are returned as a list.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        aggregator: ResultAggregator = ResultAggregator.LIST,
    ):
        self.name = name
        self.description = description
        self.aggregator = aggregator

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Slot):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        return f"Slot({self.name!r}, {self.aggregator.value})"
