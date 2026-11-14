"""Abstract base interfaces for LLM, Embedding, and Reranker providers."""

from __future__ import annotations

from typing import Any, Protocol
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = "default"


class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        ...


class EmbeddingProvider(Protocol):
    dim: int
    backend: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class RerankProvider(Protocol):
    backend: str
    available: bool

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        """Returns sorted list of (original_document_index, score)."""
        ...
