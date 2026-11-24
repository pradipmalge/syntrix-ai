
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageCreate(BaseModel):
    role: Literal["system", "user", "assistant"]

    content: str = Field(
        min_length=1,
    )

    provider: str | None = Field(
        default=None,
        max_length=50,
    )

    model: str | None = Field(
        default=None,
        max_length=100,
    )

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    created_at: datetime
