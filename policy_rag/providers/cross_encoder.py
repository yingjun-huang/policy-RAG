"""Local Cross-Encoder reranker provider with lexical fallback."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from policy_rag.config import Settings
from policy_rag.providers.base import RerankProvider
from policy_rag.providers.fake import LexicalRerankProvider

logger = logging.getLogger(__name__)


class CrossEncoderRerankProvider:
    """Joint query-document cross-encoder rerank provider (BAAI/bge-reranker-base)."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model
        self._fallback = LexicalRerankProvider()
        self._load_lock = Lock()
        self._load_error: str | None = None

    @property
    def backend(self) -> str:
        return "cross-encoder" if self.available else "lexical"

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    def _should_load(self) -> bool:
        if not self._settings.cross_encoder_enabled:
            return False
        if self._settings.is_testing and not self._settings.cross_encoder_force:
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
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading Cross-Encoder model: {self._settings.cross_encoder_model}")
                self._model = CrossEncoder(
                    self._settings.cross_encoder_model,
                    device=self._settings.cross_encoder_device,
                    max_length=self._settings.cross_encoder_max_length,
                )
                return self._model
            except Exception as e:
                self._load_error = str(e)
                logger.warning(f"Failed to load Cross-Encoder, falling back to lexical: {e}")
                return None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        model = self._ensure_model()
        if model is None or not documents:
            return await self._fallback.rerank(query, documents, top_n=top_n)

        try:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs, show_progress_bar=False)
            scored = [(idx, float(score)) for idx, score in enumerate(scores)]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_n]
        except Exception as e:
            logger.error(f"Cross-Encoder inference failed, falling back to lexical: {e}")
            return await self._fallback.rerank(query, documents, top_n=top_n)
