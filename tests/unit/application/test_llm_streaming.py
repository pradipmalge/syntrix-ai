import pytest

from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMMessage, LLMRequest

from unittest.mock import MagicMock

from syntrix_ai.domain.exceptions import LLMConnectionError


@pytest.fixture
def mock_stream_metrics(monkeypatch):
    requests = MagicMock()
    duration = MagicMock()
    ttft = MagicMock()

    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_STREAM_REQUESTS_TOTAL",
        requests,
    )
    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_STREAM_DURATION",
        duration,
    )
    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_TIME_TO_FIRST_TOKEN",
        ttft,
    )

    return requests, duration, ttft

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

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello")
        ]
    )

    chunks = [
        chunk
        async for chunk in service.stream(request)
    ]

    assert chunks == ["Hello", " ", "SyntrixAI"]


@pytest.mark.asyncio
async def test_stream_records_success_metrics(mock_stream_metrics):
    class MockProvider:
        async def stream(self, request):
            yield "Hello"
            yield " SyntrixAI"

    service = LLMService(MockProvider())

    request = LLMRequest(
        messages=[LLMMessage(role="user", content="Hello")]
    )

    chunks = [
        chunk async for chunk in service.stream(request)
    ]

    requests, duration, ttft = mock_stream_metrics

    assert chunks == ["Hello", " SyntrixAI"]

    requests.labels.assert_called_once_with(
        provider="mock",
        model="default",
        status="success",
    )

    requests.labels.return_value.inc.assert_called_once_with()
    duration.labels.return_value.observe.assert_called_once()
    ttft.labels.return_value.observe.assert_called_once()

@pytest.mark.asyncio
async def test_stream_records_failure_metrics(mock_stream_metrics):
    class MockProvider:
        async def stream(self, request):
            yield "Partial response"
            raise LLMConnectionError("Connection lost")

    service = LLMService(MockProvider())

    request = LLMRequest(
        messages=[LLMMessage(role="user", content="Hello")]
    )

    with pytest.raises(LLMConnectionError):
        async for _ in service.stream(request):
            pass

    requests, duration, ttft = mock_stream_metrics

    requests.labels.assert_called_once_with(
        provider="mock",
        model="default",
        status="error",
    )

    requests.labels.return_value.inc.assert_called_once_with()
    duration.labels.return_value.observe.assert_called_once()

    # First chunk was received before the failure.
    ttft.labels.return_value.observe.assert_called_once()