
from functools import lru_cache

from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.infrastructure.llm.provider_factory import (
    LLMProviderFactory,
)


@lru_cache
def get_llm_service() -> LLMService:
    provider = LLMProviderFactory.create()
    return LLMService(provider)
