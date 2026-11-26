
from syntrix_ai.rag.domain.interfaces import (
    EmbeddingProvider,
    VectorStore,
)
from syntrix_ai.rag.domain.models import VectorSearchResult


class RetrievalService:
    """Retrieve relevant document chunks using semantic similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        knowledge_base_id: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not tenant_id:
            raise ValueError("Tenant ID is required.")

        if not knowledge_base_id:
            raise ValueError("Knowledge base ID is required.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        # 1. Convert the user query into an embedding.
        query_embedding = await self.embedding_provider.embed_query(
            query.strip()
        )

        # 2. Search Qdrant using the query embedding.
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return results
