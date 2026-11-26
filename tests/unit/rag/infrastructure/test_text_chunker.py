
import pytest

from syntrix_ai.rag.domain.models import (
    Document,
    LoadedDocument,
)
from syntrix_ai.rag.infrastructure.chunking.text_chunker import (
    RecursiveTextChunker,
)


def create_document():
    return Document(
        id="doc-001",
        tenant_id="tenant-A",
        knowledge_base_id="finance",
        filename="annual-report.txt",
        content_type="text/plain",
        metadata={"department": "finance"},
    )


def test_split_document_into_chunks():
    chunker = RecursiveTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    document = create_document()

    content = LoadedDocument(
        content=("Revenue increased by 20%. " * 20),
        metadata={"source": "annual-report.txt"},
    )

    chunks = chunker.split(document, content)

    assert len(chunks) > 1

    assert all(
        len(chunk.content) <= 100
        for chunk in chunks
    )

    assert [c.chunk_index for c in chunks] == list(
        range(len(chunks))
    )


def test_preserve_tenant_and_knowledge_base():
    chunker = RecursiveTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    document = create_document()

    content = LoadedDocument(
        content="Financial information. " * 20
    )

    chunks = chunker.split(document, content)

    assert all(c.tenant_id == "tenant-A" for c in chunks)
    assert all(
        c.knowledge_base_id == "finance"
        for c in chunks
    )


def test_preserve_metadata():
    chunker = RecursiveTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    document = create_document()

    content = LoadedDocument(
        content="Financial information. " * 10,
        metadata={"source": "annual-report.txt"},
    )

    chunks = chunker.split(document, content)

    assert chunks[0].metadata["department"] == "finance"
    assert chunks[0].metadata["source"] == "annual-report.txt"
    assert "start_char" in chunks[0].metadata
    assert "end_char" in chunks[0].metadata


def test_empty_document_returns_no_chunks():
    chunker = RecursiveTextChunker()

    chunks = chunker.split(
        create_document(),
        LoadedDocument(content="   "),
    )

    assert chunks == []


def test_chunk_ids_are_deterministic():
    chunker = RecursiveTextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    document = create_document()
    content = LoadedDocument(
        content="Revenue increased. " * 20
    )

    first = chunker.split(document, content)
    second = chunker.split(document, content)

    assert [c.id for c in first] == [c.id for c in second]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=0)
