
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse
from collections.abc import AsyncIterator


class LLMService:

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        return await self._provider.generate(request)

    async def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]:
        async for chunk in self._provider.stream(request):
            yield chunk
    
    async def health_check(self) -> bool:
        return await self._provider.health_check()
