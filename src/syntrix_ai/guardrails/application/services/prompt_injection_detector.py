
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptInjectionResult:
    is_suspicious: bool
    matched_patterns: tuple[str, ...]


class PromptInjectionDetector:
    """Detects common prompt injection patterns using rules."""

    PATTERNS = {
        "instruction_override": (
            r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
            r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b",
            r"\boverride\s+(the\s+)?system\s+prompt\b",
        ),
        "role_manipulation": (
            r"\byou\s+are\s+now\s+(a|an|the)\b",
            r"\bact\s+as\s+(a|an|the)\b",
            r"\bpretend\s+to\s+be\b",
        ),
        "prompt_extraction": (
            r"\breveal\s+(your\s+)?(system\s+)?prompt\b",
            r"\bshow\s+(me\s+)?(your\s+)?system\s+prompt\b",
            r"\bprint\s+(your\s+)?hidden\s+instructions\b",
        ),
        "instruction_bypass": (
            r"\bbypass\s+(all\s+)?(safety|security)\s+(rules|filters|checks)\b",
            r"\bdisable\s+(your\s+)?safety\s+(rules|filters)\b",
        ),
    }

    def detect(self, text: str) -> PromptInjectionResult:
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        normalized_text = " ".join(text.lower().split())

        matched_patterns = []

        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized_text):
                    matched_patterns.append(category)
                    break

        return PromptInjectionResult(
            is_suspicious=bool(matched_patterns),
            matched_patterns=tuple(matched_patterns),
        )
