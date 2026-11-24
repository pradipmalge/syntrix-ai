
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from syntrix_ai.application.services.conversation_chat_service import (
    ConversationChatService,
)


@pytest.mark.asyncio
async def test_chat_persists_user_and_assistant_messages():
    conversation_id = uuid4()
    now = datetime.now(timezone.utc)

    user_message = SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content="What is RAG?",
        provider=None,
        model=None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        created_at=now,
    )

    assistant_message = SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content="RAG stands for Retrieval-Augmented Generation.",
        provider="ollama",
        model="qwen3:4b",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        created_at=now,
    )

    conversation_service = AsyncMock()
    conversation_service.add_message.side_effect = [
        user_message,
        assistant_message,
    ]
    conversation_service.get_messages.return_value = [user_message]

    llm_service = AsyncMock()
    llm_service.generate.return_value = SimpleNamespace(
        content="RAG stands for Retrieval-Augmented Generation.",
        provider="ollama",
        model="qwen3:4b",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )

    service = ConversationChatService(
        conversation_service=conversation_service,
        llm_service=llm_service,
    )

    result = await service.chat(
        conversation_id=conversation_id,
        content="What is RAG?",
    )

    assert result is not None
    assert result.conversation_id == conversation_id

    assert result.user_message.content == "What is RAG?"
    assert result.assistant_message.content.startswith("RAG stands")

    assert conversation_service.add_message.await_count == 2
    conversation_service.get_messages.assert_awaited_once()
    llm_service.generate.assert_awaited_once()
