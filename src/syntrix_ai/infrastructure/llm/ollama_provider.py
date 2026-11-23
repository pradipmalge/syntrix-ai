
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from syntrix_ai.core.config import get_settings
from syntrix_ai.domain.interfaces.llm_provider import LLMProvider
from syntrix_ai.domain.models.llm import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class OllamaProvider(LLMProvider):

    def __init__(self):
        self.settings = get_settings()

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        role_mapping = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        messages = [
            role_mapping[message.role](
                content=message.content
            )
            for message in request.messages
        ]

        model_name = request.model or self.settings.llm_model

        model = ChatOllama(
            model=model_name,
            base_url=self.settings.ollama_base_url,
            temperature=request.temperature,
            num_predict=request.max_tokens,
            reasoning=False,
        )

        response = await model.ainvoke(messages)

        usage_metadata = response.usage_metadata or {}

        usage = TokenUsage(
            input_tokens=usage_metadata.get("input_tokens", 0),
            output_tokens=usage_metadata.get("output_tokens", 0),
            total_tokens=usage_metadata.get("total_tokens", 0),
        )

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        return LLMResponse(
            content=content,
            model=model_name,
            provider="ollama",
            usage=usage,
            metadata=response.response_metadata or {},
        )

    async def health_check(self) -> bool:
        try:
            model = ChatOllama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
            )

            await model.ainvoke(
                [HumanMessage(content="Reply with OK")]
            )

            return True

        except Exception:
            return False
