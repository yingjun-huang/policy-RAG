"""BM25 lexical retriever over a jieba-tokenized inverted index."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from policy_rag.retrieval.filters import matches_metadata_filter
from policy_rag.retrieval.tokenizer import tokenize
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk


def _enrich_chunk_text(chunk: RetrievedChunk) -> str:
    """Prepends title and section to text for context-enriched search."""
    prefix = f"[{chunk.title} - {chunk.section or '条款'}] "
    return prefix + chunk.content


class BM25Retriever:
    """Lexical inverted index retriever powered by BM25Okapi."""

    def __init__(self, corpus: list[RetrievedChunk] | None = None) -> None:
        self._chunks: list[RetrievedChunk] = []
        self._bm25: BM25Okapi | None = None
        if corpus:
            self.build(corpus)

    def build(self, corpus: list[RetrievedChunk]) -> None:
        self._chunks = list(corpus)
        if not self._chunks:
            self._bm25 = None
            return

        tokenized_corpus = [tokenize(_enrich_chunk_text(c)) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        eligible_indices = [
            i
            for i, chunk in enumerate(self._chunks)
            if matches_metadata_filter(chunk, metadata_filter)
        ]
        ranked_indices = sorted(eligible_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self._chunks[idx].model_copy(deep=True)
            chunk.score = float(scores[idx])
            chunk.rank = rank
            chunk.source = "bm25"
            results.append(chunk)

        return results

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Async alias for consistent retriever interface."""
        return self.search(query, top_k=top_k, metadata_filter=metadata_filter)
