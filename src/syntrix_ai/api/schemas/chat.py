
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from syntrix_ai.api.schemas.message import MessageResponse


class ChatRequest(BaseModel):
    content: str = Field(min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conversation_id: UUID
    user_message: MessageResponse
    assistant_message: MessageResponse
