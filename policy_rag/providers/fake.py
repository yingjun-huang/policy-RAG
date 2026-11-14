"""Deterministic fake and fallback providers for offline testing and rapid baseline verification."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from policy_rag.providers.base import LLMResponse, TokenUsage


class FakeLLMProvider:
    """Mock LLM returning canned responses or matching query keywords."""

    def __init__(self, canned: dict[str, str] | None = None, default_response: str = "这是默认的模拟回答。") -> None:
        self.canned = canned or {}
        self.default_response = default_response
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break

        for key, value in self.canned.items():
            if key in user_content:
                return LLMResponse(content=value, usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20))

        return LLMResponse(content=self.default_response, usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20))


class HashEmbeddingProvider:
    """Deterministic token-hash embedding provider for zero-weight CI/testing."""

    backend: str = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = text.lower().split() or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(len(digest), self.dim * 4), 4):
                index = struct.unpack_from(">I", digest, offset)[0] % self.dim
                vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class LexicalRerankProvider:
    """Token-overlap heuristic reranker for CI and offline testing."""

    backend: str = "lexical"
    available: bool = True

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        query_chars = set(query.lower())
        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            doc_chars = set(doc.lower())
            if not query_chars:
                score = 0.0
            else:
                score = len(query_chars & doc_chars) / len(query_chars)
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]
