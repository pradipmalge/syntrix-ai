
from functools import lru_cache
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.infrastructure.llm.provider_factory import (
    LLMProviderFactory,
)

from syntrix_ai.application.services import ConversationService
from syntrix_ai.infrastructure.persistence.database import get_db_session

from syntrix_ai.application.services.conversation_chat_service import (
    ConversationChatService,
)
from syntrix_ai.application.services.conversation_service import (
    ConversationService,
)
from syntrix_ai.application.services.llm_service import LLMService

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    provider = LLMProviderFactory.create()
    return LLMService(provider)


def get_conversation_service(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationService:
    return ConversationService(session)


def get_conversation_chat_service(
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
) -> ConversationChatService:
    return ConversationChatService(
        conversation_service=conversation_service,
        llm_service=llm_service,
    )