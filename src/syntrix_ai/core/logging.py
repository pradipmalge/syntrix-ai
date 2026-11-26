
import logging

from opentelemetry import trace

from syntrix_ai.core.config import get_settings


class TraceContextFilter(logging.Filter):
    """Inject OpenTelemetry trace context into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        span_context = trace.get_current_span().get_span_context()

        if span_context.is_valid:
            record.trace_id = format(span_context.trace_id, "032x")
            record.span_id = format(span_context.span_id, "016x")
            record.trace_sampled = span_context.trace_flags.sampled
        else:
            record.trace_id = "none"
            record.span_id = "none"
            record.trace_sampled = False

        return True


def configure_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "trace_id=%(trace_id)s | "
            "span_id=%(span_id)s | "
            "sampled=%(trace_sampled)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    trace_filter = TraceContextFilter()

    for handler in logging.getLogger().handlers:
        handler.addFilter(trace_filter)
