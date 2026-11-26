
from collections.abc import Sequence

import httpx

from syntrix_ai.core.config import get_settings
from syntrix_ai.rag.domain.exceptions import EmbeddingError
from syntrix_ai.rag.domain.interfaces import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate document and query embeddings using Ollama."""

    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self.timeout = settings.embedding_timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        return await self._embed(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Query text cannot be empty.")

        embeddings = await self._embed([text])
        return embeddings[0]

    async def _embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:
                response = await client.post(
                    url,
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                )

                response.raise_for_status()
                payload = response.json()

            embeddings = payload.get("embeddings")

            if not isinstance(embeddings, list):
                raise EmbeddingError(
                    "Ollama returned an invalid embedding response."
                )

            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    "Embedding count does not match input count."
                )

            result: list[list[float]] = []

            for vector in embeddings:
                if not isinstance(vector, list):
                    raise EmbeddingError(
                        "Ollama returned an invalid embedding vector."
                    )

                if len(vector) != self.dimension:
                    raise EmbeddingError(
                        f"Expected {self.dimension} dimensions, "
                        f"received {len(vector)}."
                    )

                result.append([float(value) for value in vector])

            return result

        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed: "
                f"HTTP {exc.response.status_code}"
            ) from exc

        except httpx.RequestError as exc:
            raise EmbeddingError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"Invalid Ollama embedding response: {exc}"
            ) from exc
