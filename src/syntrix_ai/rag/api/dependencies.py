
from collections.abc import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.api.dependencies import get_llm_service
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.infrastructure.persistence.database import (
    get_db_session,
)
from syntrix_ai.rag.application.services.ingestion_service import (
    DocumentIngestionService,
)
from syntrix_ai.rag.application.services.loader_registry import (
    DocumentLoaderRegistry,
)
from syntrix_ai.rag.domain.repositories import DocumentRepository
from syntrix_ai.rag.infrastructure.chunking.text_chunker import (
    RecursiveTextChunker,
)
from syntrix_ai.rag.infrastructure.embeddings.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)
from syntrix_ai.rag.infrastructure.persistence.document_repository import (
    SqlAlchemyDocumentRepository,
)
from syntrix_ai.rag.infrastructure.vectorstores.qdrant_vector_store import (
    QdrantVectorStore,
)

from syntrix_ai.rag.application.services.retrieval_service import (
    RetrievalService,
)

from syntrix_ai.rag.application.services.rag_answer_service import (
    RAGAnswerService,
)

from syntrix_ai.application.services.conversation_service import (
    ConversationService,
)
from syntrix_ai.application.services.conversation_context_service import (
    ConversationContextService,
)
from syntrix_ai.application.services.question_rewriter_service import (
    QuestionRewriterService,
)

def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    return SqlAlchemyDocumentRepository(session)


async def get_vector_store() -> AsyncIterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore()

    try:
        yield vector_store
    finally:
        await vector_store.close()


def get_embedding_provider() -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider()


def get_document_ingestion_service(
    document_repository: DocumentRepository = Depends(
        get_document_repository
    ),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_provider: OllamaEmbeddingProvider = Depends(
        get_embedding_provider
    ),
) -> DocumentIngestionService:

    return DocumentIngestionService(
        loader_registry=DocumentLoaderRegistry(),
        chunker=RecursiveTextChunker(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        document_repository=document_repository,
    )

def get_retrieval_service(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_provider: OllamaEmbeddingProvider = Depends(
        get_embedding_provider
    ),
) -> RetrievalService:
    return RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

def get_conversation_service(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationService:
    return ConversationService(session)

def get_conversation_context_service(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationContextService:
    conversation_service = ConversationService(session)

    return ConversationContextService(
        conversation_service=conversation_service,
    )

def get_rag_answer_service(
    retrieval_service: RetrievalService = Depends(
        get_retrieval_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
    conversation_context_service: ConversationContextService = Depends(
        get_conversation_context_service
    ),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> RAGAnswerService:

    question_rewriter_service = QuestionRewriterService(
        llm_service=llm_service,
    )

    return RAGAnswerService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        conversation_context_service=conversation_context_service,
        question_rewriter_service=question_rewriter_service,
        conversation_service=conversation_service,
    )

