
from io import BytesIO

from pypdf import PdfReader

from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.domain.interfaces import DocumentLoader
from syntrix_ai.rag.domain.models import LoadedDocument


class PdfDocumentLoader(DocumentLoader):

    async def load(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> LoadedDocument:
        try:
            reader = PdfReader(BytesIO(content))

            pages = []
            extracted_text = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    pages.append({
                        "page_number": page_number,
                        "content": text,
                    })
                    extracted_text.append(text)

            if not extracted_text:
                raise DocumentProcessingError(
                    f"No extractable text found in {filename}."
                )

            return LoadedDocument(
                content="\n\n".join(extracted_text),
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "page_count": len(reader.pages),
                },
                pages=pages,
            )

        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to extract PDF {filename}: {exc}"
            ) from exc
