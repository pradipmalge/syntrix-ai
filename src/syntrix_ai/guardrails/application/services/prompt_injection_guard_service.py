
from dataclasses import dataclass

from syntrix_ai.guardrails.application.services.prompt_injection_detector import (
    PromptInjectionDetector,
)


class PromptInjectionBlockedError(ValueError):
    """Raised when potentially malicious input is blocked."""


@dataclass(frozen=True)
class PromptInjectionGuardResult:
    allowed: bool
    matched_patterns: tuple[str, ...]


class PromptInjectionGuardService:
    """Enforces prompt injection detection policies."""

    def __init__(
        self,
        detector: PromptInjectionDetector | None = None,
    ) -> None:
        self._detector = detector or PromptInjectionDetector()

    def inspect(self, text: str) -> PromptInjectionGuardResult:
        detection = self._detector.detect(text)

        return PromptInjectionGuardResult(
            allowed=not detection.is_suspicious,
            matched_patterns=detection.matched_patterns,
        )

    def validate(self, text: str) -> None:
        result = self.inspect(text)

        if not result.allowed:
            raise PromptInjectionBlockedError(
                "Input blocked by prompt injection security policy."
            )
