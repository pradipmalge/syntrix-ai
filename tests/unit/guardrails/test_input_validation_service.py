
import pytest

from syntrix_ai.guardrails.application.services.input_validation_service import (
    InputValidationError,
    InputValidationService,
)


@pytest.fixture
def validator():
    return InputValidationService(max_length=100)


def test_validates_normal_text(validator):
    result = validator.validate_text("  Hello SyntrixAI  ")

    assert result == "Hello SyntrixAI"


def test_rejects_empty_input(validator):
    with pytest.raises(InputValidationError):
        validator.validate_text("   ")


def test_rejects_input_exceeding_maximum_length(validator):
    with pytest.raises(InputValidationError):
        validator.validate_text("A" * 101)


def test_rejects_null_bytes(validator):
    with pytest.raises(InputValidationError):
        validator.validate_text("Hello\x00World")


def test_rejects_invalid_control_characters(validator):
    with pytest.raises(InputValidationError):
        validator.validate_text("Hello\x01World")


def test_allows_newlines_and_tabs(validator):
    result = validator.validate_text("Hello\nWorld\t!")

    assert result == "Hello\nWorld\t!"


def test_supports_custom_maximum_length(validator):
    result = validator.validate_text(
        "Hello",
        max_length=10,
    )

    assert result == "Hello"
