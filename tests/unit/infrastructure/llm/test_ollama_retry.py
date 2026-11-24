from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from syntrix_ai.domain.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
)
from syntrix_ai.domain.models.llm import LLMMessage, LLMRequest
from syntrix_ai.infrastructure.llm.ollama_provider import OllamaProvider


@pytest.fixture
def provider():
    instance = OllamaProvider()
    instance.settings.llm_max_retries = 2
    instance.settings.llm_retry_base_delay_seconds = 1.0
    return instance


@pytest.mark.asyncio
async def test_retries_connection_failure_then_succeeds(provider):
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=[
            LLMConnectionError("Connection refused"),
            LLMConnectionError("Connection refused"),
            "success",
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await provider._invoke_with_retry(model, [])

    assert result == "success"
    assert model.ainvoke.await_count == 3
    assert sleep.await_count == 2
    sleep.assert_any_await(1.0)
    sleep.assert_any_await(2.0)


@pytest.mark.asyncio
async def test_does_not_retry_generation_error(provider):
    from syntrix_ai.domain.exceptions import LLMGenerationError

    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=LLMGenerationError("Generation failed")
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(LLMGenerationError):
            await provider._invoke_with_retry(model, [])

    assert model.ainvoke.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_raises_after_max_retries(provider):
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=LLMTimeoutError("Request timed out")
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMTimeoutError):
            await provider._invoke_with_retry(model, [])

    assert model.ainvoke.await_count == 3

@pytest.mark.asyncio
async def test_stream_retries_before_first_token(provider):
    class FakeModel:
        def __init__(self, error=None, chunks=None):
            self.error = error
            self.chunks = chunks or []

        async def astream(self, messages):
            if self.error:
                raise self.error

            for chunk in self.chunks:
                yield SimpleNamespace(content=chunk)

    models = [
        FakeModel(error=LLMConnectionError("Connection refused")),
        FakeModel(chunks=["Hello", " SyntrixAI"]),
    ]

    with (
        patch.object(provider, "_build_model", side_effect=models) as build,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = [
            chunk async for chunk in provider.stream(
                LLMRequest(messages=[LLMMessage(role="user", content="Hi")])
            )
        ]

    assert result == ["Hello", " SyntrixAI"]
    assert build.call_count == 2


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_first_token(provider):
    class FakeModel:
        async def astream(self, messages):
            yield SimpleNamespace(content="Partial response")
            raise LLMConnectionError("Connection lost")

    with (
        patch.object(provider, "_build_model", return_value=FakeModel()) as build,
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        with pytest.raises(LLMConnectionError):
            async for _ in provider.stream(
                LLMRequest(messages=[LLMMessage(role="user", content="Hi")])
            ):
                pass

    assert build.call_count == 1
    sleep.assert_not_awaited()