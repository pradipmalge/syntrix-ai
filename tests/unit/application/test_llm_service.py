
from unittest.mock import AsyncMock, MagicMock

import pytest

from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


@pytest.fixture
def llm_request():
    return LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello")
        ]
    )


@pytest.fixture
def mock_metrics(monkeypatch):
    requests = MagicMock()
    duration = MagicMock()
    tokens = MagicMock()

    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_REQUESTS_TOTAL",
        requests,
    )
    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_REQUEST_DURATION",
        duration,
    )
    monkeypatch.setattr(
        "syntrix_ai.application.services.llm_service.LLM_TOKEN_USAGE",
        tokens,
    )

    return requests, duration, tokens


@pytest.mark.asyncio
async def test_generate_records_success_metrics(
    llm_request, mock_metrics
):
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            content="Hello!",
            model="qwen3:4b",
            provider="ollama",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )
    )

    service = LLMService(provider)

    response = await service.generate(llm_request)

    requests, duration, tokens = mock_metrics

    assert response.content == "Hello!"

    requests.labels.assert_called_once_with(
        provider="ollama",
        model="qwen3:4b",
        status="success",
    )

    requests.labels.return_value.inc.assert_called_once_with()

    duration.labels.return_value.observe.assert_called_once()

    assert tokens.labels.call_count == 2


@pytest.mark.asyncio
async def test_generate_records_failure_metrics(
    llm_request, mock_metrics
):
    provider = MagicMock()
    provider.generate = AsyncMock(
        side_effect=RuntimeError("Provider unavailable")
    )

    service = LLMService(provider)

    with pytest.raises(RuntimeError):
        await service.generate(llm_request)

    requests, duration, tokens = mock_metrics

    requests.labels.assert_called_once_with(
        provider="magicmock",
        model="default",
        status="error",
    )

    requests.labels.return_value.inc.assert_called_once_with()

    duration.labels.return_value.observe.assert_called_once()

    tokens.labels.assert_not_called()
