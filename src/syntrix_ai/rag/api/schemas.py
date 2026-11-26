
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from syntrix_ai.rag.domain.models import DocumentStatus


class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    status: DocumentStatus
    source_uri: str | None = None
    metadata: dict = {}
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DocumentIngestionResponse(DocumentResponse):
    chunk_count: int
    embedding_dimension: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int

class RetrievalRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class RetrievedChunkResponse(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunkResponse]
    total: int


class RAGAskRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class RAGSourceResponse(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSourceResponse]
    total_sources: int
    model: str
    provider: str