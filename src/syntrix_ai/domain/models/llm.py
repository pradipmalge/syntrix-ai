
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMRequest(BaseModel):
    messages: list[LLMMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: TokenUsage
    metadata: dict[str, Any] = Field(default_factory=dict)
