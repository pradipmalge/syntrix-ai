
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from syntrix_ai.guardrails.application.services.input_validation_service import (
    InputValidationError,
    InputValidationService,
)
from syntrix_ai.guardrails.application.services.prompt_injection_guard_service import (
    PromptInjectionGuardService,
)
from syntrix_ai.rag.api.dependencies import get_rag_answer_service
from syntrix_ai.rag.api.schemas import (
    RAGAskRequest,
    RAGAskResponse,
    RAGSourceResponse,
)
from syntrix_ai.rag.application.services.rag_answer_service import (
    RAGAnswerService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["RAG Answer Generation"],
)

input_validator = InputValidationService(max_length=12_000)
prompt_injection_guard = PromptInjectionGuardService()


@router.post(
    "/{knowledge_base_id}/ask",
    response_model=RAGAskResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_knowledge_base(
    knowledge_base_id: str,
    request: RAGAskRequest,
    rag_service: RAGAnswerService = Depends(
        get_rag_answer_service
    ),
):
    # Step 1: Validate the question.
    try:
        validated_question = input_validator.validate_text(
            request.question,
            field_name="question",
        )

    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Step 2: Inspect for prompt injection.
    detection = prompt_injection_guard.inspect(validated_question)

    if not detection.allowed:
        logger.warning(
            "RAG prompt injection blocked: knowledge_base_id=%s categories=%s",
            knowledge_base_id,
            detection.matched_patterns,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request blocked by prompt injection security policy.",
        )

    # Step 3: Execute RAG with validated input.
    try:
        result = await rag_service.answer(
            question=validated_question,
            tenant_id=request.tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            conversation_id=request.conversation_id,
        )

        return RAGAskResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                RAGSourceResponse(
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    chunk_index=source.chunk_index,
                    score=source.score,
                    content=source.content,
                    metadata=source.metadata,
                )
                for source in result.sources
            ],
            total_sources=result.total_sources,
            model=result.model,
            provider=result.provider,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "RAG answer generation failed: knowledge_base_id=%s",
            knowledge_base_id,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAG answer generation failed. Check application logs.",
        ) from exc
