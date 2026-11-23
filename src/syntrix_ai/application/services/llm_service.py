
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse


class LLMService:

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        return await self._provider.generate(request)

    async def health_check(self) -> bool:
        return await self._provider.health_check()
