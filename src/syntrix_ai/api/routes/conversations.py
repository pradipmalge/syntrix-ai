
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from syntrix_ai.api.dependencies import get_conversation_service
from syntrix_ai.api.schemas import (
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
)
from syntrix_ai.application.services import ConversationService


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.create_conversation(request.title)


@router.get(
    "",
    response_model=list[ConversationResponse],
)
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.list_conversations(
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    conversation = await service.get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    deleted = await service.delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_message(
    conversation_id: UUID,
    request: MessageCreate,
    service: ConversationService = Depends(get_conversation_service),
):
    message = await service.add_message(
        conversation_id=conversation_id,
        role=request.role,
        content=request.content,
        provider=request.provider,
        model=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=request.total_tokens,
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return message


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def get_messages(
    conversation_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: ConversationService = Depends(get_conversation_service),
):
    messages = await service.get_messages(
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
    )

    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return messages
