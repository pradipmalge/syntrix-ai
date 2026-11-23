
from abc import ABC, abstractmethod

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
    async def health_check(self) -> bool:
        """Check whether the provider is reachable."""
        raise NotImplementedError
