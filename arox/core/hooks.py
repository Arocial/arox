from typing import TYPE_CHECKING, Protocol

from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import DeferredToolRequests

if TYPE_CHECKING:
    from arox.core.llm_base import LLMBaseAgent


class PreStepHook(Protocol):
    async def __call__(
        self, agent: "LLMBaseAgent", input_content: str | None
    ) -> None: ...


class PostStepHook(Protocol):
    async def __call__(
        self,
        agent: "LLMBaseAgent",
        input_content: str | None,
        result: AgentRunResult[DeferredToolRequests | str] | None,
    ) -> None: ...


class OnFailureHook(Protocol):
    def __call__(
        self,
        agent: "LLMBaseAgent",
        input_content: str | None,
        new_messages: list[ModelMessage],
    ) -> None: ...
