
import logging

from fastapi import APIRouter, Depends, HTTPException

from syntrix_ai.api.dependencies import get_llm_service
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMRequest, LLMResponse
from syntrix_ai.guardrails.application.services.input_validation_service import (
    InputValidationError,
    InputValidationService,
)
from syntrix_ai.guardrails.application.services.prompt_injection_guard_service import (
    PromptInjectionGuardService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])

input_validator = InputValidationService(max_length=12_000)
prompt_injection_guard = PromptInjectionGuardService()


@router.post("/generate", response_model=LLMResponse)
async def generate(
    request: LLMRequest,
    service: LLMService = Depends(get_llm_service),
):
    # Step 1: Validate incoming messages.
    try:
        validated_messages = []

        for message in request.messages:
            validated_content = input_validator.validate_text(
                message.content,
                field_name=f"{message.role} message",
            )

            validated_messages.append(
                message.model_copy(
                    update={"content": validated_content}
                )
            )

    except InputValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Step 2: Inspect messages for prompt injection.
    for message in validated_messages:
        detection = prompt_injection_guard.inspect(message.content)

        if not detection.allowed:
            logger.warning(
                "Prompt injection blocked: role=%s categories=%s",
                message.role,
                detection.matched_patterns,
            )

            raise HTTPException(
                status_code=400,
                detail="Request blocked by prompt injection security policy.",
            )

    # Step 3: Create a validated request.
    validated_request = request.model_copy(
        update={"messages": validated_messages}
    )

    # Step 4: Send the request to the LLM.
    try:
        return await service.generate(validated_request)

    except Exception as exc:
        logger.exception("LLM generation failed")

        raise HTTPException(
            status_code=502,
            detail="LLM provider request failed",
        ) from exc
