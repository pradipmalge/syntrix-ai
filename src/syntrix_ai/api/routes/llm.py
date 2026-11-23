
import logging

from fastapi import APIRouter, Depends, HTTPException

from syntrix_ai.api.dependencies import get_llm_service
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post("/generate", response_model=LLMResponse)
async def generate(
    request: LLMRequest,
    service: LLMService = Depends(get_llm_service),
):
    try:
        return await service.generate(request)

    except Exception as exc:
        logger.exception("LLM generation failed")

        raise HTTPException(
            status_code=502,
            detail="LLM provider request failed",
        ) from exc
