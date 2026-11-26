
from uuid import UUID
from collections.abc import AsyncIterator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from syntrix_ai.application.services.conversation_service import (
    ConversationService,
)
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMMessage, LLMRequest
from syntrix_ai.api.schemas.chat import ChatResponse


tracer = trace.get_tracer(
    "syntrix_ai.application.conversation_chat"
)


class ConversationChatService:

    def __init__(
        self,
        conversation_service: ConversationService,
        llm_service: LLMService,
    ):
        self._conversation_service = conversation_service
        self._llm_service = llm_service

    async def chat(
        self,
        conversation_id: UUID,
        content: str,
    ) -> ChatResponse | None:

        with tracer.start_as_current_span(
            "conversation.chat"
        ) as span:

            span.set_attribute(
                "syntrix.conversation.id",
                str(conversation_id),
            )
            span.set_attribute(
                "syntrix.conversation.operation",
                "chat",
            )

            try:
                # 1. Persist user message
                with tracer.start_as_current_span(
                    "conversation.save_user_message"
                ):
                    user_message = await self._conversation_service.add_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=content,
                    )

                if user_message is None:
                    span.set_attribute(
                        "syntrix.conversation.result",
                        "not_found",
                    )
                    return None

                # 2. Load conversation history
                with tracer.start_as_current_span(
                    "conversation.load_history"
                ):
                    history = await self._conversation_service.get_messages(
                        conversation_id=conversation_id,
                        limit=50,
                        offset=0,
                    )

                if history is None:
                    span.set_attribute(
                        "syntrix.conversation.result",
                        "not_found",
                    )
                    return None

                span.set_attribute(
                    "syntrix.conversation.message_count",
                    len(history),
                )

                # 3. Convert history into LLM messages
                with tracer.start_as_current_span(
                    "conversation.prepare_llm_messages"
                ):
                    llm_messages = [
                        LLMMessage(
                            role=message.role,
                            content=message.content,
                        )
                        for message in history
                    ]

                # 4. Generate response
                llm_response = await self._llm_service.generate(
                    LLMRequest(messages=llm_messages)
                )

                # 5. Persist assistant response
                with tracer.start_as_current_span(
                    "conversation.save_assistant_message"
                ):
                    assistant_message = await self._conversation_service.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=llm_response.content,
                        provider=llm_response.provider,
                        model=llm_response.model,
                        input_tokens=llm_response.usage.input_tokens,
                        output_tokens=llm_response.usage.output_tokens,
                        total_tokens=llm_response.usage.total_tokens,
                    )

                if assistant_message is None:
                    span.set_attribute(
                        "syntrix.conversation.result",
                        "assistant_persistence_failed",
                    )
                    span.set_status(
                        Status(
                            StatusCode.ERROR,
                            "Failed to persist assistant message",
                        )
                    )
                    return None

                span.set_attribute(
                    "syntrix.conversation.result",
                    "success",
                )
                span.set_status(Status(StatusCode.OK))

                # 6. Return chat response
                return ChatResponse(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                )

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise

    async def stream_chat(
        self,
        conversation_id: UUID,
        content: str,
    ) -> AsyncIterator[str]:

        with tracer.start_as_current_span(
            "conversation.stream_chat"
        ) as span:

            span.set_attribute(
                "syntrix.conversation.id",
                str(conversation_id),
            )
            span.set_attribute(
                "syntrix.conversation.operation",
                "stream",
            )

            response_chunks = []

            try:
                # 1. Persist user message
                with tracer.start_as_current_span(
                    "conversation.save_user_message"
                ):
                    user_message = await self._conversation_service.add_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=content,
                    )

                if user_message is None:
                    raise ValueError("Conversation not found")

                # 2. Load conversation history
                with tracer.start_as_current_span(
                    "conversation.load_history"
                ):
                    history = await self._conversation_service.get_messages(
                        conversation_id=conversation_id,
                        limit=50,
                        offset=0,
                    )

                if history is None:
                    raise ValueError("Conversation not found")

                span.set_attribute(
                    "syntrix.conversation.message_count",
                    len(history),
                )

                # 3. Convert history into LLM messages
                with tracer.start_as_current_span(
                    "conversation.prepare_llm_messages"
                ):
                    llm_messages = [
                        LLMMessage(
                            role=message.role,
                            content=message.content,
                        )
                        for message in history
                    ]

                llm_request = LLMRequest(messages=llm_messages)

                # 4. Stream response and accumulate content
                async for chunk in self._llm_service.stream(llm_request):
                    if chunk:
                        response_chunks.append(chunk)
                        yield chunk

                assistant_content = "".join(response_chunks)

                if not assistant_content:
                    raise RuntimeError("LLM returned an empty response")

                # 5. Persist completed assistant response
                with tracer.start_as_current_span(
                    "conversation.save_assistant_message"
                ):
                    await self._conversation_service.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_content,
                    )

                span.set_attribute(
                    "syntrix.conversation.response_chunks",
                    len(response_chunks),
                )
                span.set_status(Status(StatusCode.OK))

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise

    async def conversation_exists(
        self,
        conversation_id: UUID,
    ) -> bool:

        with tracer.start_as_current_span(
            "conversation.exists"
        ) as span:

            span.set_attribute(
                "syntrix.conversation.id",
                str(conversation_id),
            )

            try:
                with tracer.start_as_current_span(
                    "conversation.load_messages"
                ):
                    messages = await self._conversation_service.get_messages(
                        conversation_id=conversation_id,
                        limit=1,
                        offset=0,
                    )

                exists = messages is not None

                span.set_attribute(
                    "syntrix.conversation.exists",
                    exists,
                )
                span.set_status(Status(StatusCode.OK))

                return exists

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise
