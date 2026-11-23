
from syntrix_ai.core.config import get_settings
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.infrastructure.llm.ollama_provider import OllamaProvider


class LLMProviderFactory:

    @staticmethod
    def create() -> LLMProvider:

        settings = get_settings()

        provider = settings.llm_provider.lower()

        if provider == "ollama":
            return OllamaProvider()

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )
