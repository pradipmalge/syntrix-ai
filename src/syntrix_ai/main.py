
from contextlib import asynccontextmanager

from fastapi import FastAPI

from syntrix_ai.api.router import api_router
from syntrix_ai.api.routes.metrics import router as metrics_router
from syntrix_ai.core.config import get_settings
from syntrix_ai.core.logging import configure_logging
from syntrix_ai.infrastructure.observability.tracing import (
    configure_tracing,
    shutdown_tracing,
)

from syntrix_ai.infrastructure.persistence.database import (
    configure_database_instrumentation, configure_database_pool_metrics,
)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    # Initialize OpenTelemetry
    configure_tracing()
    configure_database_instrumentation()
    configure_database_pool_metrics()
    yield

    # Flush pending spans during shutdowncls
    shutdown_tracing()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise AI Application Platform",
    lifespan=lifespan,
)

if settings.otel_enabled:
    FastAPIInstrumentor.instrument_app(app)

app.include_router(metrics_router)

app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/")
async def root():
    return {
        "application": settings.app_name,
        "message": "Welcome to SyntrixAI",
        "docs": "/docs",
    }
