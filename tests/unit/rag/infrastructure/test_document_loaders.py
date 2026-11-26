
from io import BytesIO

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from syntrix_ai.rag.domain.exceptions import DocumentProcessingError
from syntrix_ai.rag.infrastructure.loaders.pdf_loader import (
    PdfDocumentLoader,
)
from syntrix_ai.rag.infrastructure.loaders.docx_loader import (
    DocxDocumentLoader,
)
from syntrix_ai.rag.infrastructure.loaders.text_loader import (
    TextDocumentLoader,
)


@pytest.mark.asyncio
async def test_text_loader():
    loader = TextDocumentLoader()

    result = await loader.load(
        b"SyntrixAI RAG pipeline",
        "test.txt",
        "text/plain",
    )

    assert result.content == "SyntrixAI RAG pipeline"
    assert result.metadata["filename"] == "test.txt"


@pytest.mark.asyncio
async def test_text_loader_empty_content():
    loader = TextDocumentLoader()

    with pytest.raises(DocumentProcessingError):
        await loader.load(b" ", "empty.txt", "text/plain")


@pytest.mark.asyncio
async def test_docx_loader_extracts_paragraphs_and_tables():
    buffer = BytesIO()
    document = DocxDocument()

    document.add_paragraph("SyntrixAI architecture")

    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Component"
    table.cell(0, 1).text = "Technology"

    document.save(buffer)

    loader = DocxDocumentLoader()

    result = await loader.load(
        buffer.getvalue(),
        "test.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert "SyntrixAI architecture" in result.content
    assert "Component | Technology" in result.content
    assert result.metadata["table_count"] == 1


@pytest.mark.asyncio
async def test_pdf_loader_empty_pdf():
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)

    loader = PdfDocumentLoader()

    with pytest.raises(DocumentProcessingError):
        await loader.load(
            buffer.getvalue(),
            "empty.pdf",
            "application/pdf",
        )


@pytest.mark.asyncio
async def test_pdf_loader_invalid_content():
    loader = PdfDocumentLoader()

    with pytest.raises(DocumentProcessingError):
        await loader.load(
            b"not a valid PDF",
            "invalid.pdf",
            "application/pdf",
        )
