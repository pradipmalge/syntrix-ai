class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class LLMConnectionError(LLMProviderError):
    """Raised when the provider cannot be reached."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM request times out."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when provider authentication fails."""


class LLMGenerationError(LLMProviderError):
    """Raised when the provider fails to generate a response."""