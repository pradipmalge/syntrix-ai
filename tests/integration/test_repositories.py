
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from syntrix_ai.core.config import get_settings
from syntrix_ai.infrastructure.persistence.repositories import (
    ConversationRepository,
    MessageRepository,
)


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(
        get_settings().database_url,
        poolclass=NullPool,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    try:
        async with session_factory() as session:
            yield session
            await session.rollback()
    finally:
        await engine.dispose()



@pytest.mark.asyncio
async def test_create_and_retrieve_conversation(db_session: AsyncSession):
    repository = ConversationRepository(db_session)

    conversation = await repository.create("Integration test")

    retrieved = await repository.get_by_id(conversation.id)

    assert retrieved is not None
    assert retrieved.id == conversation.id
    assert retrieved.title == "Integration test"


@pytest.mark.asyncio
async def test_create_and_retrieve_messages(db_session: AsyncSession):
    conversation_repository = ConversationRepository(db_session)
    message_repository = MessageRepository(db_session)

    conversation = await conversation_repository.create("Chat test")

    await message_repository.create(
        conversation_id=conversation.id,
        role="user",
        content="Hello SyntrixAI",
    )

    await message_repository.create(
        conversation_id=conversation.id,
        role="assistant",
        content="Hello! How can I help?",
        provider="ollama",
        model="qwen3:4b",
    )

    messages = await message_repository.list_by_conversation(
        conversation.id
    )

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_message_count(db_session: AsyncSession):
    conversation_repository = ConversationRepository(db_session)
    message_repository = MessageRepository(db_session)

    conversation = await conversation_repository.create("Count test")

    await message_repository.create(
        conversation.id,
        "user",
        "First message",
    )

    await message_repository.create(
        conversation.id,
        "assistant",
        "Second message",
    )

    count = await message_repository.count_by_conversation(
        conversation.id
    )

    assert count == 2


@pytest.mark.asyncio
async def test_delete_conversation(db_session: AsyncSession):
    conversation_repository = ConversationRepository(db_session)
    message_repository = MessageRepository(db_session)

    conversation = await conversation_repository.create("Delete test")

    await message_repository.create(
        conversation.id,
        "user",
        "Temporary message",
    )

    deleted = await conversation_repository.delete(conversation.id)

    assert deleted is True

    retrieved = await conversation_repository.get_by_id(
        conversation.id
    )

    assert retrieved is None
