
from collections.abc import AsyncIterator

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from syntrix_ai.core.config import get_settings
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

import httpx
import asyncio

from syntrix_ai.domain.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMProviderError,
    LLMTimeoutError,
)

class OllamaProvider(LLMProvider):

    def __init__(self):
        self.settings = get_settings()

    def _convert_messages(self, request: LLMRequest):
        role_mapping = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        messages = []

        for message in request.messages:
            message_class = role_mapping.get(message.role)

            if message_class is None:
                raise ValueError(
                    f"Unsupported message role: {message.role}"
                )

            messages.append(
                message_class(content=message.content)
            )

        return messages

    def _build_model(self, request: LLMRequest) -> ChatOllama:
        return ChatOllama(
            model=request.model or self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=(
                request.temperature
                if request.temperature is not None
                else self.settings.llm_temperature
            ),
            num_predict=(
                request.max_tokens
                if request.max_tokens is not None
                else self.settings.llm_max_tokens
            ),
            reasoning=self.settings.llm_reasoning,
            client_kwargs={
                "timeout": self.settings.llm_timeout_seconds,
            },
        )

    async def _invoke_with_retry(self, model, messages):
        settings = self.settings

        for attempt in range(settings.llm_max_retries + 1):
            try:
                return await model.ainvoke(messages)

            except LLMProviderError as exc:
                provider_error = exc

            except Exception as exc:
                provider_error = self._map_provider_error(exc)

            retryable = isinstance(
                provider_error,
                (LLMConnectionError, LLMTimeoutError),
            )

            if not retryable or attempt >= settings.llm_max_retries:
                raise provider_error

            delay = (
                settings.llm_retry_base_delay_seconds
                * (2 ** attempt)
            )

            await asyncio.sleep(delay)

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        try:
            messages = self._convert_messages(request)
            model = self._build_model(request)

            response = await self._invoke_with_retry(model, messages)

            usage_metadata = response.usage_metadata or {}

            usage = TokenUsage(
                input_tokens=usage_metadata.get("input_tokens", 0),
                output_tokens=usage_metadata.get("output_tokens", 0),
                total_tokens=usage_metadata.get("total_tokens", 0),
            )

            content = response.content

            if not isinstance(content, str):
                content = str(content)

            return LLMResponse(
                content=content,
                model=request.model or self.settings.llm_model,
                provider="ollama",
                usage=usage,
                metadata=response.response_metadata or {},
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise self._map_provider_error(exc) from exc

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = self._convert_messages(request)

        for attempt in range(self.settings.llm_max_retries + 1):
            yielded_any = False

            try:
                model = self._build_model(request)

                async for chunk in model.astream(messages):
                    content = chunk.content

                    if isinstance(content, str) and content:
                        yielded_any = True
                        yield content

                return

            except LLMProviderError as exc:
                provider_error = exc

            except Exception as exc:
                provider_error = self._map_provider_error(exc)

            retryable = isinstance(
                provider_error,
                (LLMConnectionError, LLMTimeoutError),
            )

            if (
                not retryable
                or yielded_any
                or attempt >= self.settings.llm_max_retries
            ):
                raise provider_error

            delay = (
                self.settings.llm_retry_base_delay_seconds
                * (2 ** attempt)
            )

            await asyncio.sleep(delay)

    async def health_check(self) -> bool:
        try:
            model = ChatOllama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
            )

            await model.ainvoke(
                [HumanMessage(content="Reply with OK")]
            )

            return True

        except Exception:
            return False

    def _map_provider_error(self, exc: Exception) -> LLMProviderError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError("Ollama request timed out.")

        if isinstance(exc, httpx.TransportError):
            return LLMConnectionError(
                "Unable to connect to the Ollama server."
            )

        status_code = getattr(exc, "status_code", None)

        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        if status_code in (401, 403):
            return LLMAuthenticationError(
                "Ollama authentication failed."
            )

        return LLMGenerationError(
            f"Ollama generation failed: {exc}"
        )