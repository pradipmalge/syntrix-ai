
import pytest

from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class FakeLLMProvider(LLMProvider):

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content="Test response",
            model="fake-model",
            provider="fake",
            usage=TokenUsage(),
        )

    async def stream(self, request):
        for chunk in ["Hello", " ", "SyntrixAI"]:
            yield chunk

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_llm_service_generate():

    service = LLMService(FakeLLMProvider())

    request = LLMRequest(
        messages=[
            {
                "role": "user",
                "content": "Hello SyntrixAI",
            }
        ]
    )

    response = await service.generate(request)

    assert response.content == "Test response"
    assert response.provider == "fake"
