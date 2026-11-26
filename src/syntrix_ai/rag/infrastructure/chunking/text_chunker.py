
import re
from uuid import NAMESPACE_URL, uuid5

from syntrix_ai.core.config import get_settings
from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.domain.interfaces import TextChunker
from syntrix_ai.rag.domain.models import (
    Document,
    DocumentChunk,
    LoadedDocument,
)


class RecursiveTextChunker(TextChunker):
    """Split documents into overlapping, boundary-aware chunks."""

    _BOUNDARY_PATTERN = re.compile(
        r"\n\n|\n|(?<=[.!?])\s+| "
    )

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.rag_chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.rag_chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

    def split(
        self,
        document: Document,
        content: LoadedDocument,
    ) -> list[DocumentChunk]:

        text = content.content.strip()

        if not text:
            return []

        chunks: list[DocumentChunk] = []

        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            hard_end = min(
                start + self.chunk_size,
                text_length,
            )

            end = hard_end

            if hard_end < text_length:
                minimum_boundary = (
                    start + self.chunk_size // 2
                )

                boundaries = list(
                    self._BOUNDARY_PATTERN.finditer(
                        text,
                        minimum_boundary,
                        hard_end,
                    )
                )

                if boundaries:
                    end = boundaries[-1].end()

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"syntrix:{document.id}:{chunk_index}",
                    )
                )

                metadata = {
                    **document.metadata,
                    **content.metadata,
                    "start_char": start,
                    "end_char": end,
                }

                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        tenant_id=document.tenant_id,
                        knowledge_base_id=(
                            document.knowledge_base_id
                        ),
                        content=chunk_text,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )

                chunk_index += 1

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                raise DocumentProcessingError(
                    "Chunking failed to make forward progress."
                )

            start = next_start

        return chunks
