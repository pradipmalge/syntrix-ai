
from syntrix_ai.application.services.llm_service import LLMService
from syntrix_ai.domain.models.llm import LLMMessage, LLMRequest


class QuestionRewriterService:
    """Converts conversational follow-up questions into standalone queries."""

    SYSTEM_PROMPT = """
You are a search query rewriting assistant.

Your task is to rewrite the user's latest question into a
standalone question that can be understood without conversation history.

Instructions:
1. Use conversation history to understand references such as
   "it", "that", "the second one", and "those requirements".
2. Preserve the original intent of the user's question.
3. Include relevant entities, technologies, and context from history.
4. Do not answer the question.
5. Do not introduce facts that are not supported by the conversation.
6. If the question is already self-contained, keep its meaning unchanged.
7. Return only the rewritten question. No explanation or additional text.
"""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    async def rewrite(
        self,
        question: str,
        conversation_history: list[LLMMessage],
    ) -> str:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not conversation_history:
            return question.strip()

        messages = [
            LLMMessage(
                role="system",
                content=self.SYSTEM_PROMPT,
            ),
            *conversation_history,
            LLMMessage(
                role="user",
                content=(
                    "Rewrite the following latest question into "
                    "a standalone search query:\n\n"
                    f"{question.strip()}"
                ),
            ),
        ]

        response = await self._llm_service.generate(
            LLMRequest(
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
        )

        rewritten_question = response.content.strip()

        if not rewritten_question:
            raise RuntimeError(
                "Question rewriting returned an empty result."
            )

        return rewritten_question
