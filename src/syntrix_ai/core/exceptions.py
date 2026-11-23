
class SyntrixAIException(Exception):
    """Base exception for application-level errors."""


class ConfigurationError(SyntrixAIException):
    pass


class ProviderError(SyntrixAIException):
    pass


class ResourceNotFoundError(SyntrixAIException):
    pass


class ExternalServiceError(SyntrixAIException):
    pass
