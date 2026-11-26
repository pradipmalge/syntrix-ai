
import logging
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from syntrix_ai.rag.api.dependencies import (
    get_document_ingestion_service,
    get_document_repository,
    get_vector_store,
)
from syntrix_ai.rag.api.schemas import (
    DocumentIngestionResponse,
    DocumentListResponse,
    DocumentResponse,
)
from syntrix_ai.rag.application.services.ingestion_service import (
    DocumentIngestionService,
)
from syntrix_ai.rag.domain.exceptions import (
    UnsupportedDocumentTypeError,
)
from syntrix_ai.rag.domain.models import Document, DocumentStatus
from syntrix_ai.rag.domain.repositories import DocumentRepository
from syntrix_ai.rag.domain.interfaces import VectorStore
from syntrix_ai.rag.infrastructure.loaders.pdf_loader import (
    PdfDocumentLoader,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    knowledge_base_id: str = Form(...),
    ingestion_service: DocumentIngestionService = Depends(
        get_document_ingestion_service
    ),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    content = await file.read(MAX_DOCUMENT_SIZE + 1)

    if len(content) > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Document exceeds the 10 MB upload limit.",
        )

    content_type = file.content_type or "application/octet-stream"

    document = Document(
        id=str(uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        filename=file.filename,
        content_type=content_type,
        status=DocumentStatus.PENDING,
    )

    try:
        # Validate the file type before creating its database record.
        ingestion_service.loader_registry.get_loader(
            document.filename,
            document.content_type,
        )

        result = await ingestion_service.ingest(
            document=document,
            content=content,
        )

        persisted_document = await ingestion_service.document_repository.get_by_id(
            UUID(document.id),
            tenant_id,
        )

        if persisted_document is None:
            raise RuntimeError(
                "Document was ingested but its database record "
                "could not be retrieved."
            )

        return DocumentIngestionResponse(
            id=persisted_document.id,
            tenant_id=persisted_document.tenant_id,
            knowledge_base_id=persisted_document.knowledge_base_id,
            filename=persisted_document.filename,
            content_type=persisted_document.content_type,
            status=persisted_document.status,
            source_uri=persisted_document.source_uri,
            metadata=persisted_document.metadata,
            created_at=persisted_document.created_at,
            chunk_count=result.chunk_count,
            embedding_dimension=result.embedding_dimension,
        )

    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=415,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Document upload failed: document_id=%s",
            document.id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check application logs.",
        ) from exc

    finally:
        await file.close()


@router.get(
    "",
    response_model=DocumentListResponse,
)
async def list_documents(
    tenant_id: str = Query(...),
    knowledge_base_id: str = Query(...),
    repository: DocumentRepository = Depends(
        get_document_repository
    ),
):
    documents = await repository.list_by_knowledge_base(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )

    documents = [
        document
        for document in documents
        if document.status != DocumentStatus.DELETED
    ]

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=document.id,
                tenant_id=document.tenant_id,
                knowledge_base_id=document.knowledge_base_id,
                filename=document.filename,
                content_type=document.content_type,
                status=document.status,
                source_uri=document.source_uri,
                metadata=document.metadata,
                created_at=document.created_at,
            )
            for document in documents
        ],
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    document_id: UUID,
    tenant_id: str = Query(...),
    repository: DocumentRepository = Depends(
        get_document_repository
    ),
):
    document = await repository.get_by_id(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    if (
        document is None
        or document.status == DocumentStatus.DELETED
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentResponse(
        id=document.id,
        tenant_id=document.tenant_id,
        knowledge_base_id=document.knowledge_base_id,
        filename=document.filename,
        content_type=document.content_type,
        status=document.status,
        source_uri=document.source_uri,
        metadata=document.metadata,
        created_at=document.created_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: UUID,
    tenant_id: str = Query(...),
    repository: DocumentRepository = Depends(
        get_document_repository
    ),
    vector_store: VectorStore = Depends(get_vector_store),
):
    document = await repository.get_by_id(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    if (
        document is None
        or document.status == DocumentStatus.DELETED
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        # Remove vectors first. If this fails, retain the DB record.
        await vector_store.delete_document(
            document_id=str(document_id),
            tenant_id=tenant_id,
        )

        # Soft-delete the PostgreSQL record.
        await repository.update_status(
            document_id=document_id,
            tenant_id=tenant_id,
            status=DocumentStatus.DELETED.value,
        )

    except Exception as exc:
        logger.exception(
            "Document deletion failed: document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
        ) from exc
