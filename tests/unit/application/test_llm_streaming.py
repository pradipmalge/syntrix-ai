from unittest.mock import AsyncMock

import pytest

from syntrix_ai.application.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_llm_service_streams_chunks():
    class MockProvider:
        async def generate(self, request):
            raise NotImplementedError

        async def health_check(self):
            return True

        async def stream(self, request):
            for chunk in ["Hello", " ", "SyntrixAI"]:
                yield chunk

    service = LLMService(MockProvider())

    chunks = [
        chunk
        async for chunk in service.stream(None)
    ]

    assert chunks == ["Hello", " ", "SyntrixAI"]