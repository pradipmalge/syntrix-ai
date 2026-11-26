
from uuid import UUID
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.infrastructure.persistence.models.document import (
    DocumentRecord,
)
from syntrix_ai.rag.domain.models import (
    Document,
    DocumentStatus,
)
from syntrix_ai.rag.domain.repositories import DocumentRepository


class SqlAlchemyDocumentRepository(DocumentRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _to_domain(record: DocumentRecord) -> Document:
        return Document(
            id=str(record.id),
            tenant_id=record.tenant_id,
            knowledge_base_id=record.knowledge_base_id,
            filename=record.filename,
            content_type=record.content_type,
            status=DocumentStatus(record.status),
            source_uri=record.source_uri,
            metadata=record.document_metadata or {},
            created_at=record.created_at,
        )

    async def create(self, document: Document) -> Document:
        record = DocumentRecord(
            id=UUID(document.id),
            tenant_id=document.tenant_id,
            knowledge_base_id=document.knowledge_base_id,
            filename=document.filename,
            content_type=document.content_type,
            source_uri=document.source_uri,
            status=document.status.value,
            document_metadata=document.metadata,
        )

        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)

        return self._to_domain(record)

    async def get_by_id(
        self,
        document_id: UUID,
        tenant_id: str,
    ) -> Document | None:
        statement = select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == tenant_id,
        )

        result = await self.session.execute(statement)
        record = result.scalar_one_or_none()

        return self._to_domain(record) if record else None

    async def list_by_knowledge_base(
        self,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> list[Document]:
        statement = (
            select(DocumentRecord)
            .where(
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.knowledge_base_id == knowledge_base_id,
            )
            .order_by(DocumentRecord.created_at.desc())
        )

        result = await self.session.execute(statement)

        return [
            self._to_domain(record)
            for record in result.scalars().all()
        ]

    async def update_status(
        self,
        document_id: UUID,
        tenant_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        statement = (
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(
                status=status,
                error_message=error_message,
                updated_at=func.clock_timestamp(),
            )
        )

        await self.session.execute(statement)
        await self.session.flush()

    async def delete(
        self,
        document_id: UUID,
        tenant_id: str,
    ) -> bool:
        statement = (
            delete(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
        )

        result = await self.session.execute(statement)
        await self.session.flush()

        return result.rowcount > 0

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()