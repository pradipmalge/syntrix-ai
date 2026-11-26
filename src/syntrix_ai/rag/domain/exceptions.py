
class RAGError(Exception):
    """Base exception for RAG operations."""


class DocumentProcessingError(RAGError):
    """Raised when document extraction or processing fails."""


class UnsupportedDocumentTypeError(RAGError):
    """Raised when the document format is not supported."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGError):
    """Raised when vector storage or retrieval fails."""


class KnowledgeBaseNotFoundError(RAGError):
    """Raised when the requested knowledge base does not exist."""
