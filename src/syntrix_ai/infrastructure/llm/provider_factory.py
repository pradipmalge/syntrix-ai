
from syntrix_ai.core.config import get_settings
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.infrastructure.llm.ollama_provider import OllamaProvider


class LLMProviderFactory:

    _providers: dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider,
    }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[LLMProvider],
    ) -> None:
        cls._providers[name.strip().lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider_name: str | None = None,
    ) -> LLMProvider:

        settings = get_settings()

        selected_provider = (
            provider_name or settings.llm_provider
        ).strip().lower()

        provider_class = cls._providers.get(selected_provider)

        if provider_class is None:
            available = ", ".join(sorted(cls._providers))

            raise ValueError(
                f"Unsupported LLM provider: '{selected_provider}'. "
                f"Available providers: {available}"
            )

        return provider_class()
