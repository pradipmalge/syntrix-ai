
import logging

from fastapi import APIRouter, Depends, HTTPException, status

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
    try:
        result = await rag_service.answer(
            question=request.question,
            tenant_id=request.tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
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
