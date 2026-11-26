
import asyncio
import time
from collections.abc import AsyncIterator

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from syntrix_ai.core.config import get_settings
from syntrix_ai.domain.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMProviderError,
    LLMTimeoutError,
)
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


tracer = trace.get_tracer("syntrix_ai.infrastructure.ollama")


class OllamaProvider(LLMProvider):

    def __init__(self):
        self.settings = get_settings()

    def _model_name(self, request: LLMRequest) -> str:
        return request.model or self.settings.llm_model

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
            model=self._model_name(request),
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
        parent_span = trace.get_current_span()

        for attempt in range(settings.llm_max_retries + 1):

            with tracer.start_as_current_span(
                "ollama.generate.attempt"
            ) as attempt_span:

                attempt_span.set_attribute(
                    "syntrix.llm.attempt",
                    attempt + 1,
                )

                try:
                    response = await model.ainvoke(messages)
                    attempt_span.set_status(Status(StatusCode.OK))
                    return response

                except LLMProviderError as exc:
                    provider_error = exc
                    attempt_span.record_exception(exc)
                    attempt_span.set_status(
                        Status(StatusCode.ERROR, str(exc))
                    )

                except Exception as exc:
                    provider_error = self._map_provider_error(exc)
                    attempt_span.record_exception(exc)
                    attempt_span.set_status(
                        Status(StatusCode.ERROR, str(provider_error))
                    )

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

            parent_span.add_event(
                "ollama.retry",
                {
                    "syntrix.llm.attempt": attempt + 1,
                    "syntrix.llm.retry_delay_seconds": delay,
                    "syntrix.llm.error_type": type(provider_error).__name__,
                },
            )

            await asyncio.sleep(delay)

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        model_name = self._model_name(request)

        with tracer.start_as_current_span(
            "ollama.generate"
        ) as span:

            span.set_attribute("syntrix.llm.provider", "ollama")
            span.set_attribute("syntrix.llm.model", model_name)
            span.set_attribute("syntrix.llm.operation", "generate")
            span.set_attribute(
                "syntrix.llm.max_retries",
                self.settings.llm_max_retries,
            )

            start_time = time.perf_counter()

            try:
                messages = self._convert_messages(request)
                model = self._build_model(request)

                response = await self._invoke_with_retry(
                    model,
                    messages,
                )

                usage_metadata = response.usage_metadata or {}

                usage = TokenUsage(
                    input_tokens=usage_metadata.get("input_tokens", 0),
                    output_tokens=usage_metadata.get("output_tokens", 0),
                    total_tokens=usage_metadata.get("total_tokens", 0),
                )

                content = response.content

                if not isinstance(content, str):
                    content = str(content)

                span.set_attribute(
                    "syntrix.llm.input_tokens",
                    usage.input_tokens,
                )
                span.set_attribute(
                    "syntrix.llm.output_tokens",
                    usage.output_tokens,
                )
                span.set_attribute(
                    "syntrix.llm.total_tokens",
                    usage.total_tokens,
                )

                span.set_status(Status(StatusCode.OK))

                return LLMResponse(
                    content=content,
                    model=model_name,
                    provider="ollama",
                    usage=usage,
                    metadata=response.response_metadata or {},
                )

            except LLMProviderError as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise

            except Exception as exc:
                provider_error = self._map_provider_error(exc)

                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(provider_error))
                )

                raise provider_error from exc

            finally:
                span.set_attribute(
                    "syntrix.llm.duration_seconds",
                    time.perf_counter() - start_time,
                )

    async def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]:

        model_name = self._model_name(request)
        start_time = time.perf_counter()

        with tracer.start_as_current_span(
            "ollama.stream"
        ) as span:

            span.set_attribute("syntrix.llm.provider", "ollama")
            span.set_attribute("syntrix.llm.model", model_name)
            span.set_attribute("syntrix.llm.operation", "stream")
            span.set_attribute(
                "syntrix.llm.max_retries",
                self.settings.llm_max_retries,
            )

            first_chunk_time = None
            total_chunks = 0

            try:
                messages = self._convert_messages(request)

                for attempt in range(self.settings.llm_max_retries + 1):

                    yielded_any = False
                    provider_error = None

                    with tracer.start_as_current_span(
                        "ollama.stream.attempt"
                    ) as attempt_span:

                        attempt_span.set_attribute(
                            "syntrix.llm.attempt",
                            attempt + 1,
                        )

                        try:
                            model = self._build_model(request)

                            async for chunk in model.astream(messages):

                                content = chunk.content

                                if isinstance(content, str) and content:
                                    if first_chunk_time is None:
                                        first_chunk_time = time.perf_counter()

                                        span.add_event(
                                            "ollama.first_chunk_received"
                                        )

                                    yielded_any = True
                                    total_chunks += 1

                                    yield content

                            attempt_span.set_status(
                                Status(StatusCode.OK)
                            )

                            return

                        except LLMProviderError as exc:
                            provider_error = exc

                        except Exception as exc:
                            provider_error = self._map_provider_error(exc)

                        attempt_span.record_exception(provider_error)
                        attempt_span.set_status(
                            Status(
                                StatusCode.ERROR,
                                str(provider_error),
                            )
                        )

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

                    span.add_event(
                        "ollama.retry",
                        {
                            "syntrix.llm.attempt": attempt + 1,
                            "syntrix.llm.retry_delay_seconds": delay,
                            "syntrix.llm.error_type": type(provider_error).__name__,
                        },
                    )

                    await asyncio.sleep(delay)

                span.set_status(Status(StatusCode.OK))

            except LLMProviderError as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise

            except Exception as exc:
                provider_error = self._map_provider_error(exc)

                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(provider_error))
                )

                raise provider_error from exc

            finally:
                duration = time.perf_counter() - start_time

                span.set_attribute(
                    "syntrix.llm.duration_seconds",
                    duration,
                )
                span.set_attribute(
                    "syntrix.llm.response_chunks",
                    total_chunks,
                )

                if first_chunk_time is not None:
                    span.set_attribute(
                        "syntrix.llm.time_to_first_chunk_seconds",
                        first_chunk_time - start_time,
                    )

    async def health_check(self) -> bool:

        with tracer.start_as_current_span(
            "ollama.health_check"
        ) as span:

            span.set_attribute("syntrix.llm.provider", "ollama")
            span.set_attribute(
                "syntrix.llm.model",
                self.settings.llm_model,
            )

            try:
                model = ChatOllama(
                    model=self.settings.llm_model,
                    base_url=self.settings.ollama_base_url,
                )

                await model.ainvoke(
                    [HumanMessage(content="Reply with OK")]
                )

                span.set_status(Status(StatusCode.OK))
                return True

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                return False

    def _map_provider_error(
        self,
        exc: Exception,
    ) -> LLMProviderError:

        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                "Ollama request timed out."
            )

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
