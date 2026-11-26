
import time
from collections.abc import AsyncIterator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse
from syntrix_ai.infrastructure.observability.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION,
    LLM_TOKEN_USAGE,
    LLM_STREAM_REQUESTS_TOTAL,
    LLM_STREAM_DURATION,
    LLM_TIME_TO_FIRST_TOKEN,
)


tracer = trace.get_tracer("syntrix_ai.application.llm")


class LLMService:

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        provider = self._provider
        provider_name = (
            provider.__class__.__name__.replace("Provider", "").lower()
        )
        model_name = request.model or "default"

        start_time = time.perf_counter()
        status = "success"

        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("syntrix.llm.provider", provider_name)
            span.set_attribute("syntrix.llm.model", model_name)
            span.set_attribute("syntrix.llm.operation", "generate")

            try:
                response = await provider.generate(request)

                provider_name = response.provider
                model_name = response.model

                span.set_attribute("syntrix.llm.provider", provider_name)
                span.set_attribute("syntrix.llm.model", model_name)

                LLM_TOKEN_USAGE.labels(
                    provider=response.provider,
                    model=response.model,
                    token_type="input",
                ).inc(response.usage.input_tokens)

                LLM_TOKEN_USAGE.labels(
                    provider=response.provider,
                    model=response.model,
                    token_type="output",
                ).inc(response.usage.output_tokens)

                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as exc:
                status = "error"

                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )

                raise

            finally:
                duration = time.perf_counter() - start_time

                LLM_REQUESTS_TOTAL.labels(
                    provider=provider_name,
                    model=model_name,
                    status=status,
                ).inc()

                LLM_REQUEST_DURATION.labels(
                    provider=provider_name,
                    model=model_name,
                ).observe(duration)

    async def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]:

        provider_name = (
            self._provider.__class__.__name__
            .replace("Provider", "")
            .lower()
        )
        model_name = request.model or "default"

        start_time = time.perf_counter()
        first_token_time = None
        status = "success"

        with tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute("syntrix.llm.provider", provider_name)
            span.set_attribute("syntrix.llm.model", model_name)
            span.set_attribute("syntrix.llm.operation", "stream")

            try:
                async for chunk in self._provider.stream(request):

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                        span.add_event("llm.first_chunk_received")

                    yield chunk

                span.set_status(Status(StatusCode.OK))

            except Exception as exc:
                status = "error"

                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )

                raise

            finally:
                duration = time.perf_counter() - start_time

                LLM_STREAM_REQUESTS_TOTAL.labels(
                    provider=provider_name,
                    model=model_name,
                    status=status,
                ).inc()

                LLM_STREAM_DURATION.labels(
                    provider=provider_name,
                    model=model_name,
                ).observe(duration)

                if first_token_time is not None:
                    LLM_TIME_TO_FIRST_TOKEN.labels(
                        provider=provider_name,
                        model=model_name,
                    ).observe(first_token_time - start_time)

    async def health_check(self) -> bool:
        return await self._provider.health_check()
