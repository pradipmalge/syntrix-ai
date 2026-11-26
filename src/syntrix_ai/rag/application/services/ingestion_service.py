
import logging
from uuid import UUID

from syntrix_ai.rag.application.dto.ingestion import IngestionResult
from syntrix_ai.rag.application.services.loader_registry import (
    DocumentLoaderRegistry,
)
from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.domain.interfaces import (
    EmbeddingProvider,
    TextChunker,
    VectorStore,
)
from syntrix_ai.rag.domain.models import (
    Document,
    DocumentStatus,
)
from syntrix_ai.rag.domain.repositories import DocumentRepository


logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Orchestrate document extraction, embedding and indexing."""

    def __init__(
        self,
        loader_registry: DocumentLoaderRegistry,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        document_repository: DocumentRepository,
    ) -> None:
        self.loader_registry = loader_registry
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.document_repository = document_repository

    async def ingest(
        self,
        document: Document,
        content: bytes,
    ) -> IngestionResult:

        logger.info(
            "Starting document ingestion: document_id=%s tenant_id=%s",
            document.id,
            document.tenant_id,
        )

        # Persist initial document record.
        document.status = DocumentStatus.PENDING
        await self.document_repository.create(document)

        # Persist initial document record.
        await self.document_repository.commit()
        try:
            # Transition: pending -> processing
            await self.document_repository.update_status(
                document_id=UUID(document.id),
                tenant_id=document.tenant_id,
                status=DocumentStatus.PROCESSING.value,
            )
            await self.document_repository.commit()

            # 1. Resolve document loader.
            loader = self.loader_registry.get_loader(
                document.filename,
                document.content_type,
            )

            # 2. Extract document content.
            loaded_document = await loader.load(
                content=content,
                filename=document.filename,
                content_type=document.content_type,
            )

            # 3. Generate document chunks.
            chunks = self.chunker.split(
                document=document,
                content=loaded_document,
            )

            if not chunks:
                raise DocumentProcessingError(
                    f"No chunks generated for {document.filename}."
                )

            # 4. Generate embeddings.
            embeddings = await self.embedding_provider.embed_documents(
                [chunk.content for chunk in chunks]
            )

            if len(embeddings) != len(chunks):
                raise DocumentProcessingError(
                    "Embedding count does not match chunk count."
                )

            # 5. Index vectors in Qdrant.
            await self.vector_store.upsert(
                chunks=chunks,
                embeddings=embeddings,
            )

            # Transition: processing -> indexed
            await self.document_repository.update_status(
                document_id=UUID(document.id),
                tenant_id=document.tenant_id,
                status=DocumentStatus.INDEXED.value,
            )

            await self.document_repository.commit()

            logger.info(
                "Document ingestion completed: document_id=%s chunks=%d",
                document.id,
                len(chunks),
            )

            return IngestionResult(
                document_id=document.id,
                filename=document.filename,
                chunk_count=len(chunks),
                embedding_dimension=self.embedding_provider.dimension,
            )

        except Exception as exc:
            logger.exception(
                "Document ingestion failed: document_id=%s",
                document.id,
            )

            try:
                # Clear any failed or incomplete transaction.
                await self.document_repository.rollback()

                await self.document_repository.update_status(
                    document_id=UUID(document.id),
                    tenant_id=document.tenant_id,
                    status=DocumentStatus.FAILED.value,
                    error_message=str(exc)[:2000],
                )

                await self.document_repository.commit()

            except Exception:
                logger.exception(
                    "Could not persist failed ingestion status: "
                    "document_id=%s",
                    document.id,
                )

            raise
