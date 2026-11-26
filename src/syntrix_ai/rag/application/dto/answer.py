from typing import Any

from pydantic import BaseModel, Field


class RAGSource(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAnswerResult(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]
    total_sources: int
    model: str
    provider: str