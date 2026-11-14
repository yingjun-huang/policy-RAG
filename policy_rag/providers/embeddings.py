"""Embedding providers: local BGE sentence-transformers and deterministic hash fallback."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from policy_rag.config import Settings
from policy_rag.providers.base import EmbeddingProvider
from policy_rag.providers.fake import HashEmbeddingProvider

logger = logging.getLogger(__name__)


class BgeEmbeddingProvider:
    """Local dense embeddings via sentence-transformers (BAAI/bge-small-zh-v1.5)."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model
        self._load_lock = Lock()
        self._load_error: str | None = None
        self._fallback = HashEmbeddingProvider(dim=settings.embedding_dim)
        self.dim = settings.embedding_dim
        self.backend = "bge-local"

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    def _should_load(self) -> bool:
        if not self._settings.local_embedding_enabled:
            return False
        if self._settings.is_testing and not self._settings.local_embedding_force:
            return False
        return True

    def _ensure_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self._should_load():
            return None

        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                return None

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading local BGE embedding model: {self._settings.local_embedding_model}")
                self._model = SentenceTransformer(
                    self._settings.local_embedding_model,
                    device=self._settings.local_embedding_device,
                )
                dim = getattr(self._model, "get_sentence_embedding_dimension", lambda: None)()
                if dim:
                    self.dim = int(dim)
                return self._model
            except Exception as e:
                self._load_error = str(e)
                logger.warning(f"Failed to load BGE model, falling back to hash embeddings: {e}")
                return None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model is None:
            return await self._fallback.embed(texts)

        try:
            # Normalize embeddings directly for fast cosine similarity via dot product
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Dense embedding inference failed, falling back to hash: {e}")
            return await self._fallback.embed(texts)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory creating appropriate embedding provider based on environment."""
    if settings.is_testing and not settings.local_embedding_force:
        return HashEmbeddingProvider(dim=settings.embedding_dim)
    return BgeEmbeddingProvider(settings)
