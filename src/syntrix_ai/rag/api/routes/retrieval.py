import logging

from fastapi import APIRouter, Depends, HTTPException, status

from syntrix_ai.rag.api.dependencies import get_retrieval_service
from syntrix_ai.rag.api.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from syntrix_ai.rag.application.services.retrieval_service import (
    RetrievalService,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["Knowledge Base Retrieval"],
)


@router.post(
    "/{knowledge_base_id}/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def search_knowledge_base(
    knowledge_base_id: str,
    request: RetrievalRequest,
    retrieval_service: RetrievalService = Depends(
        get_retrieval_service
    ),
):
    try:
        results = await retrieval_service.retrieve(
            query=request.query,
            tenant_id=request.tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        return RetrievalResponse(
            query=request.query,
            results=[
                RetrievedChunkResponse(
                    document_id=result.chunk.document_id,
                    chunk_id=result.chunk.id,
                    content=result.chunk.content,
                    score=result.score,
                    chunk_index=result.chunk.chunk_index,
                    metadata=result.chunk.metadata,
                )
                for result in results
            ],
            total=len(results),
        )

    except Exception as exc:
        logger.exception(
            "Knowledge base retrieval failed: knowledge_base_id=%s",
            knowledge_base_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge base search failed. Check application logs.",
        ) from exc