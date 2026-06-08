from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
R = TypeVar("R")


class ResultAggregator(Enum):
    """Strategy for aggregating results from multiple slot providers."""

    DISCARD = "discard"
    FIRST = "first"
    LIST = "list"


@runtime_checkable
class Provider(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class BaseSlot[ProviderT: Provider, ReturnT]:
    """Base class for all slots."""

    def __init__(
        self,
        name: str,
        description: str = "",
        aggregator: ResultAggregator = ResultAggregator.LIST,
    ):
        self.name = name
        self.description = description
        self.aggregator = aggregator

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, BaseSlot):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, {self.aggregator.value})"


class ListSlot[ProviderT: Provider, R](BaseSlot[ProviderT, list[R]]):
    """A slot that returns a list of all provider results."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, aggregator=ResultAggregator.LIST)


class FirstSlot[ProviderT: Provider, R](BaseSlot[ProviderT, R | None]):
    """A slot that returns the result of the first provider."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, aggregator=ResultAggregator.FIRST)


class DiscardSlot[ProviderT: Provider](BaseSlot[ProviderT, None]):
    """A slot that discards all provider results (event channel)."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, aggregator=ResultAggregator.DISCARD)
