"""Application configuration powered by pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge_jd"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_name: str = "policy-rag"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    knowledge_dir: str = str(_DEFAULT_KNOWLEDGE_DIR)

    # Generic OpenAI-compatible LLM (takes precedence over DeepSeek defaults when set)
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # LLM (DeepSeek fallback)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    # Dense Embedding (BGE)
    local_embedding_enabled: bool = True
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    local_embedding_device: str = "cpu"
    local_embedding_force: bool = False
    embedding_dim: int = 512

    # Cross-Encoder Reranker (BGE)
    cross_encoder_enabled: bool = True
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    cross_encoder_device: str = "cpu"
    cross_encoder_max_length: int = 512
    cross_encoder_force: bool = False
    hf_endpoint: str = "https://hf-mirror.com"

    # Qdrant Vector Store & Scalar Payload Indexing
    vector_store_backend: str = "qdrant"
    qdrant_url: str = ""
    qdrant_path: str = ""
    qdrant_collection: str = "policy_chunks"

    # Retrieval & Self-RAG Hyperparameters
    retrieval_top_k_simple: int = 3
    retrieval_top_k_complex: int = 6
    rrf_k: int = 60
    rerank_top_n: int = 5
    max_retrieval_hops: int = 2
    parent_expansion_enabled: bool = True
    max_parent_chars: int = 1500
    max_context_chars: int = 4500
    default_as_of: str = "2024-06-01T12:00:00+08:00"

    @property
    def is_testing(self) -> bool:
        return self.app_env.lower() == "testing"

    @property
    def active_llm_api_key(self) -> str:
        return self.llm_api_key.strip() or self.deepseek_api_key.strip()

    @property
    def active_llm_base_url(self) -> str:
        return self.llm_base_url.strip() or self.deepseek_base_url.strip()

    @property
    def active_llm_model(self) -> str:
        return self.llm_model.strip() or self.deepseek_model.strip()

    @property
    def llm_configured(self) -> bool:
        return bool(self.active_llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
