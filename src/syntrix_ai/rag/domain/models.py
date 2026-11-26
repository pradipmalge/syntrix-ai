
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class Document:
    id: str
    tenant_id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class DocumentChunk:
    id: str
    document_id: str
    tenant_id: str
    knowledge_base_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VectorSearchResult:
    chunk: DocumentChunk
    score: float
