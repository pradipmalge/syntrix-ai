
from io import BytesIO

from docx import Document as DocxDocument

from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.domain.interfaces import DocumentLoader
from syntrix_ai.rag.domain.models import LoadedDocument


class DocxDocumentLoader(DocumentLoader):

    async def load(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> LoadedDocument:
        try:
            document = DocxDocument(BytesIO(content))

            sections = []

            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    sections.append(text)

            # Extract table contents as tab-separated rows.
            for table in document.tables:
                for row in table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]
                    sections.append(" | ".join(cells))

            extracted_text = "\n\n".join(sections)

            if not extracted_text.strip():
                raise DocumentProcessingError(
                    f"No extractable text found in {filename}."
                )

            return LoadedDocument(
                content=extracted_text,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "paragraph_count": len(document.paragraphs),
                    "table_count": len(document.tables),
                },
            )

        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to extract DOCX {filename}: {exc}"
            ) from exc
