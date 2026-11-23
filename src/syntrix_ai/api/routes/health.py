
from fastapi import APIRouter

from syntrix_ai.core.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    settings = get_settings()

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
    }
