
from abc import ABC, abstractmethod
from collections.abc import Sequence

from syntrix_ai.rag.domain.models import (
    Document,
    DocumentChunk,
    LoadedDocument,
    VectorSearchResult,
)


class DocumentLoader(ABC):

    @abstractmethod
    async def load(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> LoadedDocument:
        """Extract text and metadata from an uploaded document."""
        ...


class TextChunker(ABC):

    @abstractmethod
    def split(
        self,
        document: Document,
        content: LoadedDocument,
    ) -> list[DocumentChunk]:
        """Split extracted document content into chunks."""
        ...


class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    @abstractmethod
    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple document chunks."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        ...


class VectorStore(ABC):

    @abstractmethod
    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Insert or update document chunks and their vectors."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        tenant_id: str,
        knowledge_base_id: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for relevant chunks within a knowledge base."""
        ...

    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> None:
        """Delete all vectors belonging to a document."""
        ...
