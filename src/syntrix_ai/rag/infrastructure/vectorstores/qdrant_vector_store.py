
from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models

from syntrix_ai.core.config import get_settings
from syntrix_ai.rag.domain.exceptions import VectorStoreError
from syntrix_ai.rag.domain.interfaces import VectorStore
from syntrix_ai.rag.domain.models import (
    DocumentChunk,
    VectorSearchResult,
)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector storage with tenant isolation."""

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        settings = get_settings()

        self.collection_name = settings.qdrant_collection_name
        self.dimension = settings.embedding_dimension

        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert domain IDs into valid, deterministic UUIDs."""
        try:
            return str(UUID(chunk_id))
        except ValueError:
            return str(uuid5(NAMESPACE_URL, chunk_id))

    async def _ensure_collection(self) -> None:
        exists = await self.client.collection_exists(
            self.collection_name
        )

        if exists:
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )

        # Index fields used in tenant and document filtering.
        for field_name in (
            "tenant_id",
            "knowledge_base_id",
            "document_id",
        ):
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Chunk count and embedding count must match."
            )

        if not chunks:
            return

        try:
            for vector in embeddings:
                if len(vector) != self.dimension:
                    raise VectorStoreError(
                        f"Expected {self.dimension} dimensions, "
                        f"received {len(vector)}."
                    )

            await self._ensure_collection()

            points = []

            for chunk, embedding in zip(
                chunks, embeddings, strict=True
            ):
                points.append(
                    models.PointStruct(
                        id=self._point_id(chunk.id),
                        vector=embedding,
                        payload={
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "tenant_id": chunk.tenant_id,
                            "knowledge_base_id": (
                                chunk.knowledge_base_id
                            ),
                            "content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata,
                        },
                    )
                )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store vectors in Qdrant: {exc}"
            ) from exc

    async def search(
        self,
        query_embedding: list[float],
        tenant_id: str,
        knowledge_base_id: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        if len(query_embedding) != self.dimension:
            raise VectorStoreError(
                f"Expected query dimension {self.dimension}, "
                f"received {len(query_embedding)}."
            )

        if limit < 1:
            raise VectorStoreError("Search limit must be positive.")

        try:
            if not await self.client.collection_exists(
                self.collection_name
            ):
                return []

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    ),
                    models.FieldCondition(
                        key="knowledge_base_id",
                        match=models.MatchValue(
                            value=knowledge_base_id
                        ),
                    ),
                ]
            )

            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results = []

            for point in response.points:
                payload = point.payload or {}

                chunk = DocumentChunk(
                    id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    tenant_id=payload["tenant_id"],
                    knowledge_base_id=payload["knowledge_base_id"],
                    content=payload["content"],
                    chunk_index=payload["chunk_index"],
                    metadata=payload.get("metadata", {}),
                )

                results.append(
                    VectorSearchResult(
                        chunk=chunk,
                        score=point.score,
                    )
                )

            return results

        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant search failed: {exc}"
            ) from exc

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> None:
        try:
            if not await self.client.collection_exists(
                self.collection_name
            ):
                return

            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    ),
                ]
            )

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=delete_filter
                ),
                wait=True,
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document vectors: {exc}"
            ) from exc

    async def close(self) -> None:
        await self.client.close()
