
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.infrastructure.persistence.models import (
    Conversation,
    Message,
)
from syntrix_ai.infrastructure.persistence.repositories import (
    ConversationRepository,
    MessageRepository,
)


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.conversation_repository = ConversationRepository(session)
        self.message_repository = MessageRepository(session)
        self.session = session

    async def create_conversation(
        self,
        title: str | None = None,
    ) -> Conversation:
        async with self.session.begin():
            return await self.conversation_repository.create(title)

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        return await self.conversation_repository.get_by_id(
            conversation_id
        )

    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        return await self.conversation_repository.list(
            limit=limit,
            offset=offset,
        )

    async def delete_conversation(
        self,
        conversation_id: UUID,
    ) -> bool:
        async with self.session.begin():
            return await self.conversation_repository.delete(
                conversation_id
            )

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> Message | None:
        async with self.session.begin():
            conversation = (
                await self.conversation_repository.get_by_id(
                    conversation_id
                )
            )

            if conversation is None:
                return None

            return await self.message_repository.create(
                conversation_id=conversation_id,
                role=role,
                content=content,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

    async def get_messages(
        self,
        conversation_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ):
        async with self.session.begin():
            conversation = await self.conversation_repository.get_by_id(
                conversation_id
            )

            if conversation is None:
                return None

            return await self.message_repository.list_by_conversation(
                conversation_id=conversation_id,
                limit=limit,
                offset=offset,
            )
