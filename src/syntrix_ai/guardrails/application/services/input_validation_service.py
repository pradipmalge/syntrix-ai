
import unicodedata


class InputValidationError(ValueError):
    """Raised when user input fails validation."""


class InputValidationService:
    """Validates text before it enters AI processing pipelines."""

    def __init__(self, max_length: int = 12_000) -> None:
        if max_length < 1:
            raise ValueError("Maximum input length must be positive.")

        self._max_length = max_length

    def validate_text(
        self,
        value: str,
        field_name: str = "input",
        max_length: int | None = None,
    ) -> str:
        if not isinstance(value, str):
            raise InputValidationError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise InputValidationError(
                f"{field_name} cannot be empty."
            )

        effective_max_length = (
            max_length if max_length is not None else self._max_length
        )

        if effective_max_length < 1:
            raise ValueError("Maximum input length must be positive.")

        if len(normalized) > effective_max_length:
            raise InputValidationError(
                f"{field_name} exceeds the maximum "
                f"length of {effective_max_length} characters."
            )

        for character in normalized:
            if character == "\x00":
                raise InputValidationError(
                    f"{field_name} contains a null byte."
                )

            if (
                unicodedata.category(character) == "Cc"
                and character not in ("\n", "\r", "\t")
            ):
                raise InputValidationError(
                    f"{field_name} contains invalid control characters."
                )

        return normalized
