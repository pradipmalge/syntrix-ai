
from unittest.mock import AsyncMock

import pytest

from syntrix_ai.rag.application.services.retrieval_service import (
    RetrievalService,
)


@pytest.mark.asyncio
async def test_retrieve_generates_embedding_and_searches():

    embedding_provider = AsyncMock()
    vector_store = AsyncMock()

    embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]
    vector_store.search.return_value = []

    service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = await service.retrieve(
        query="What are access control requirements?",
        tenant_id="infra",
        knowledge_base_id="iso",
        top_k=5,
    )

    embedding_provider.embed_query.assert_awaited_once_with(
        "What are access control requirements?"
    )

    vector_store.search.assert_awaited_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        tenant_id="infra",
        knowledge_base_id="iso",
        limit=5,
        score_threshold=None,
    )

    assert results == []


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query():

    service = RetrievalService(
        embedding_provider=AsyncMock(),
        vector_store=AsyncMock(),
    )

    with pytest.raises(ValueError, match="Search query"):
        await service.retrieve(
            query=" ",
            tenant_id="infra",
            knowledge_base_id="iso",
        )
