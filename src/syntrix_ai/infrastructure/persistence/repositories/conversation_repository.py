
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.infrastructure.persistence.models import Conversation


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(title=title)

        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)

        return conversation

    async def get_by_id(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        statement = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def delete(
        self,
        conversation_id: UUID,
    ) -> bool:
        conversation = await self.get_by_id(conversation_id)

        if conversation is None:
            return False

        await self.session.delete(conversation)
        await self.session.flush()

        return True
