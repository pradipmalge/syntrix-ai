
from uuid import UUID

from syntrix_ai.application.services.conversation_service import (
    ConversationService,
)
from syntrix_ai.domain.models.llm import LLMMessage


class ConversationContextService:
    """Loads and prepares conversation history for AI processing."""

    def __init__(
        self,
        conversation_service: ConversationService,
    ) -> None:
        self._conversation_service = conversation_service

    async def get_context(
        self,
        conversation_id: UUID,
        limit: int = 20,
    ) -> list[LLMMessage]:

        if limit < 1:
            raise ValueError("Context limit must be greater than zero.")

        # Load recent messages from PostgreSQL.
        history = await self._conversation_service.get_messages(
            conversation_id=conversation_id,
            limit=limit,
            offset=0,
        )

        if history is None:
            raise ValueError(
                f"Conversation not found: {conversation_id}"
            )

        # Keep conversational messages, excluding old system prompts.
        context = [
            LLMMessage(
                role=message.role,
                content=message.content,
            )
            for message in history
            if message.role in {"user", "assistant"}
        ]

        return context
