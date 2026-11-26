
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from syntrix_ai.core.config import get_settings


_tracer_provider: TracerProvider | None = None


def configure_tracing() -> None:
    """Configure OpenTelemetry tracing."""

    global _tracer_provider

    settings = get_settings()

    if not settings.otel_enabled:
        return

    if _tracer_provider is not None:
        return

    resource = Resource.create(
        {
            "service.name": settings.app_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.app_env,
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_endpoint,
        insecure=True,
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def get_tracer(name: str = "syntrix_ai"):
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush and shut down the tracing provider."""

    global _tracer_provider

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
