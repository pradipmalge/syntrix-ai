
import pytest

from syntrix_ai.rag.application.services.ingestion_service import (
    DocumentIngestionService,
)
from syntrix_ai.rag.application.services.loader_registry import (
    DocumentLoaderRegistry,
)
from syntrix_ai.rag.domain.models import (
    Document,
    DocumentChunk,
    LoadedDocument,
)
from uuid import UUID

class FakeLoader:
    async def load(self, content, filename, content_type):
        return LoadedDocument(
            content="Revenue increased by 20%. " * 20
        )


class FakeRegistry:
    def get_loader(self, filename, content_type):
        return FakeLoader()


class FakeChunker:
    def split(self, document, content):
        return [
            DocumentChunk(
                id="chunk-001",
                document_id=document.id,
                tenant_id=document.tenant_id,
                knowledge_base_id=document.knowledge_base_id,
                content="Revenue increased by 20%.",
                chunk_index=0,
            )
        ]


class FakeEmbeddingProvider:
    dimension = 768

    async def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]

    async def embed_query(self, text):
        return [0.1] * 768


class FakeVectorStore:
    def __init__(self):
        self.saved_chunks = []

    async def upsert(self, chunks, embeddings):
        self.saved_chunks.extend(chunks)

    async def search(self, *args, **kwargs):
        return []

    async def delete_document(self, *args, **kwargs):
        pass


class FakeDocumentRepository:

    def __init__(self):
        self.documents = {}
        self.status_updates = []

    async def create(self, document):
        self.documents[document.id] = document
        return document

    async def update_status(
        self,
        document_id,
        tenant_id,
        status,
        error_message=None,
    ):
        self.status_updates.append(status)

        document = self.documents[str(document_id)]
        document.status = status

    async def get_by_id(self, document_id, tenant_id):
        return self.documents.get(str(document_id))

    async def list_by_knowledge_base(
        self,
        tenant_id,
        knowledge_base_id,
    ):
        return []

    async def delete(self, document_id, tenant_id):
        return False


@pytest.mark.asyncio
async def test_document_ingestion_pipeline():
    vector_store = FakeVectorStore()
    document_repository = FakeDocumentRepository()

    service = DocumentIngestionService(
        loader_registry=FakeRegistry(),
        chunker=FakeChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        document_repository=document_repository
    )

    document = Document(
        id="00000000-0000-0000-0000-000000000001",
        tenant_id="tenant-A",
        knowledge_base_id="finance",
        filename="report.txt",
        content_type="text/plain",
    )

    result = await service.ingest(
        document=document,
        content=b"test document",
    )
    assert result.document_id == document.id
    assert document_repository.status_updates == [
        "processing",
        "indexed",
    ]
    assert result.chunk_count == 1
    assert result.embedding_dimension == 768
    assert result.status == "indexed"
    assert len(vector_store.saved_chunks) == 1


def test_loader_registry_supports_pdf():
    registry = DocumentLoaderRegistry()

    loader = registry.get_loader(
        "report.pdf",
        "application/pdf",
    )

    from syntrix_ai.rag.infrastructure.loaders.pdf_loader import (
        PdfDocumentLoader,
    )

    assert isinstance(loader, PdfDocumentLoader)
