
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.infrastructure.persistence.models import (
    Conversation,
    Message,
)


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> Message:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        self.session.add(message)

        # Update conversation activity when a new message is added.
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=func.now())
        )

        await self.session.flush()
        await self.session.refresh(message)

        return message

    async def list_by_conversation(
        self,
        conversation_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_by_conversation(
        self,
        conversation_id: UUID,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )

        result = await self.session.execute(statement)
        return result.scalar_one()
