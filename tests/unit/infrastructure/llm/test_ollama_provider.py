import httpx
import pytest

from syntrix_ai.domain.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMTimeoutError,
)
from syntrix_ai.infrastructure.llm.ollama_provider import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider()


def test_maps_timeout_error(provider):
    error = httpx.TimeoutException("Request timed out")

    result = provider._map_provider_error(error)

    assert isinstance(result, LLMTimeoutError)


def test_maps_connection_error(provider):
    error = httpx.ConnectError("Connection refused")

    result = provider._map_provider_error(error)

    assert isinstance(result, LLMConnectionError)


@pytest.mark.parametrize("status_code", [401, 403])
def test_maps_authentication_error(provider, status_code):
    class ProviderHTTPError(Exception):
        pass

    error = ProviderHTTPError("Authentication failed")
    error.status_code = status_code

    result = provider._map_provider_error(error)

    assert isinstance(result, LLMAuthenticationError)


def test_maps_unknown_error_to_generation_error(provider):
    error = ValueError("Unexpected provider failure")

    result = provider._map_provider_error(error)

    assert isinstance(result, LLMGenerationError) 