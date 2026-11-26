
import pytest
from qdrant_client import AsyncQdrantClient

from syntrix_ai.rag.domain.models import (
    DocumentChunk,
)
from syntrix_ai.rag.infrastructure.vectorstores.qdrant_vector_store import (
    QdrantVectorStore,
)


@pytest.mark.asyncio
async def test_upsert_and_search():
    client = AsyncQdrantClient(location=":memory:")

    store = QdrantVectorStore(client=client)

    chunks = [
        DocumentChunk(
            id="chunk-001",
            document_id="doc-001",
            tenant_id="tenant-A",
            knowledge_base_id="finance",
            content="Annual revenue increased by 20%.",
            chunk_index=0,
        ),
        DocumentChunk(
            id="chunk-002",
            document_id="doc-002",
            tenant_id="tenant-B",
            knowledge_base_id="finance",
            content="Revenue decreased by 10%.",
            chunk_index=0,
        ),
    ]

    embeddings = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    try:
        await store.upsert(chunks, embeddings)

        results = await store.search(
            query_embedding=[0.1] * 768,
            tenant_id="tenant-A",
            knowledge_base_id="finance",
            limit=5,
        )

        assert len(results) == 1
        assert results[0].chunk.id == "chunk-001"
        assert results[0].chunk.tenant_id == "tenant-A"

    finally:
        await store.close()
