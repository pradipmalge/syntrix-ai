
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "syntrix-ai"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    api_v1_prefix: str = "/api/v1"

    llm_provider: str = "ollama"
    llm_model: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"

    # LLM generation defaults
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_reasoning: bool = False


    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2
    llm_retry_base_delay_seconds: float = 1.0

    # OpenTelemetry
    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:4317"

    # RAG settings
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_timeout_seconds: int = 60

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "syntrix_documents"

    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200

    database_url: str = (
        "postgresql+asyncpg://syntrix:syntrix_dev"
        "@localhost:5432/syntrix"
    )

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
