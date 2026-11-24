
from uuid import UUID

from syntrix_ai.application.services.conversation_service import (
    ConversationService,
)
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMMessage, LLMRequest
from syntrix_ai.api.schemas.chat import ChatResponse
from collections.abc import AsyncIterator

class ConversationChatService:

    def __init__(
        self,
        conversation_service: ConversationService,
        llm_service: LLMService,
    ):
        self._conversation_service = conversation_service
        self._llm_service = llm_service

    async def chat(
        self,
        conversation_id: UUID,
        content: str,
    ) -> ChatResponse | None:

        # 1. Persist the user's message
        user_message = await self._conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )

        if user_message is None:
            return None

        # 2. Load conversation history
        history = await self._conversation_service.get_messages(
            conversation_id=conversation_id,
            limit=50,
            offset=0,
        )

        if history is None:
            return None

        # 3. Convert history into LLM messages
        llm_messages = [
            LLMMessage(
                role=message.role,
                content=message.content,
            )
            for message in history
        ]

        # 4. Generate response
        llm_response = await self._llm_service.generate(
            LLMRequest(messages=llm_messages)
        )

        # 5. Persist assistant response
        assistant_message = await self._conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=llm_response.content,
            provider=llm_response.provider,
            model=llm_response.model,
            input_tokens=llm_response.usage.input_tokens,
            output_tokens=llm_response.usage.output_tokens,
            total_tokens=llm_response.usage.total_tokens,
        )

        if assistant_message is None:
            return None

        # 6. Return chat response
        return ChatResponse(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )


    async def stream_chat(
        self,
        conversation_id: UUID,
        content: str,
    ) -> AsyncIterator[str]:

        # 1. Persist user message
        user_message = await self._conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )

        if user_message is None:
            raise ValueError("Conversation not found")

        # 2. Load conversation history
        history = await self._conversation_service.get_messages(
            conversation_id=conversation_id,
            limit=50,
            offset=0,
        )

        if history is None:
            raise ValueError("Conversation not found")

        # 3. Convert history into LLM messages
        llm_messages = [
            LLMMessage(
                role=message.role,
                content=message.content,
            )
            for message in history
        ]

        llm_request = LLMRequest(messages=llm_messages)

        # 4. Stream response and accumulate content
        response_chunks = []

        async for chunk in self._llm_service.stream(llm_request):
            if chunk:
                response_chunks.append(chunk)
                yield chunk

        # 5. Persist completed assistant response
        assistant_content = "".join(response_chunks)

        if not assistant_content:
            raise RuntimeError("LLM returned an empty response")

        await self._conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
        )


    async def conversation_exists(self, conversation_id: UUID) -> bool:
        messages = await self._conversation_service.get_messages(
            conversation_id=conversation_id,
            limit=1,
            offset=0,
        )

        return messages is not None