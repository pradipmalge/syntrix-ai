
import logging
import json
from fastapi.responses import StreamingResponse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from syntrix_ai.api.dependencies import get_conversation_chat_service
from syntrix_ai.api.schemas.chat import ChatRequest, ChatResponse
from syntrix_ai.application.services.conversation_chat_service import (
    ConversationChatService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def chat(
    conversation_id: UUID,
    request: ChatRequest,
    service: ConversationChatService = Depends(
        get_conversation_chat_service
    ),
):
    try:
        result = await service.chat(
            conversation_id=conversation_id,
            content=request.content,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Conversation chat failed")

        raise HTTPException(
            status_code=502,
            detail="Chat generation failed",
        ) from exc



@router.post("/{conversation_id}/chat/stream")
async def stream_chat(
    conversation_id: UUID,
    request: ChatRequest,
    service: ConversationChatService = Depends(
        get_conversation_chat_service
    ),
):
    # Validate before starting the HTTP response
    exists = await service.conversation_exists(conversation_id)

    if not exists:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    async def event_generator():
        try:
            async for chunk in service.stream_chat(
                conversation_id=conversation_id,
                content=request.content,
            ):
                event_data = json.dumps({"content": chunk})
                yield f"event: token\ndata: {event_data}\n\n"

            yield (
                "event: completed\n"
                'data: {"status":"completed"}\n\n'
            )

        except Exception:
            logger.exception("Conversation streaming failed")

            yield (
                "event: error\n"
                'data: {"message":"Response generation failed"}\n\n'
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

