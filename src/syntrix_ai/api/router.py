
from fastapi import APIRouter

from syntrix_ai.api.routes.health import router as health_router
from syntrix_ai.api.routes.llm import router as llm_router


api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(llm_router)
