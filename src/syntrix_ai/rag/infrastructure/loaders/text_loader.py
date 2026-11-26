
from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.domain.interfaces import DocumentLoader
from syntrix_ai.rag.domain.models import LoadedDocument


class TextDocumentLoader(DocumentLoader):

    async def load(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> LoadedDocument:
        try:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("utf-16")

            if not text.strip():
                raise DocumentProcessingError(
                    f"Document {filename} is empty."
                )

            return LoadedDocument(
                content=text,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "encoding": "utf-8-sig or utf-16",
                },
            )

        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to extract TXT {filename}: {exc}"
            ) from exc
