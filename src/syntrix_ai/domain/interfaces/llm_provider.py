
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse


class LLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate a response using an LLM provider."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream responses from the LLM provider."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Check whether the provider is reachable."""
        raise NotImplementedError
