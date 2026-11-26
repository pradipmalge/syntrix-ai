
from abc import ABC, abstractmethod
from uuid import UUID

from syntrix_ai.rag.domain.models import Document


class DocumentRepository(ABC):

    @abstractmethod
    async def commit(self) -> None:
        ...

    @abstractmethod
    async def rollback(self) -> None:
        ...

    @abstractmethod
    async def create(self, document: Document) -> Document:
        ...

    @abstractmethod
    async def get_by_id(
        self,
        document_id: UUID,
        tenant_id: str,
    ) -> Document | None:
        ...

    @abstractmethod
    async def list_by_knowledge_base(
        self,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> list[Document]:
        ...

    @abstractmethod
    async def update_status(
        self,
        document_id: UUID,
        tenant_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def delete(
        self,
        document_id: UUID,
        tenant_id: str,
    ) -> bool:
        ...
