
from pathlib import Path

from syntrix_ai.rag.domain.exceptions import (
    UnsupportedDocumentTypeError,
)
from syntrix_ai.rag.domain.interfaces import DocumentLoader
from syntrix_ai.rag.infrastructure.loaders.docx_loader import (
    DocxDocumentLoader,
)
from syntrix_ai.rag.infrastructure.loaders.pdf_loader import (
    PdfDocumentLoader,
)
from syntrix_ai.rag.infrastructure.loaders.text_loader import (
    TextDocumentLoader,
)


class DocumentLoaderRegistry:
    """Resolve document loaders by MIME type or file extension."""

    def __init__(self) -> None:
        self._loaders: dict[str, DocumentLoader] = {
            ".pdf": PdfDocumentLoader(),
            ".docx": DocxDocumentLoader(),
            ".txt": TextDocumentLoader(),
            ".md": TextDocumentLoader(),
        }

    def get_loader(
        self,
        filename: str,
        content_type: str,
    ) -> DocumentLoader:
        extension = Path(filename).suffix.lower()

        supported_types = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/markdown": ".md",
        }

        mime_extension = supported_types.get(
            content_type.split(";")[0].strip().lower()
        )

        # Prefer the file extension when it is supported.
        resolved_extension = (
            extension
            if extension in self._loaders
            else mime_extension
        )

        loader = self._loaders.get(resolved_extension or "")

        if loader is None:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {filename}"
            )

        return loader
