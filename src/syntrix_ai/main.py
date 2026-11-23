
from contextlib import asynccontextmanager

from fastapi import FastAPI

from syntrix_ai.api.router import api_router
from syntrix_ai.core.config import get_settings
from syntrix_ai.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise AI Application Platform",
    lifespan=lifespan,
)

app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/")
async def root():
    return {
        "application": settings.app_name,
        "message": "Welcome to NexusAI",
        "docs": "/docs",
    }
