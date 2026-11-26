
from dataclasses import dataclass


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    chunk_count: int
    embedding_dimension: int
    status: str = "indexed"
