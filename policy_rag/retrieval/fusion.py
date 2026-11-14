"""Reciprocal Rank Fusion (RRF) for combining multi-route retrieval candidates."""

from __future__ import annotations

from policy_rag.schemas.retrieval import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuses multiple ranked lists using the Cormack Reciprocal Rank Fusion algorithm.
    Score = sum_m 1 / (k + rank_m(chunk)).
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive")

    scores: dict[str, float] = {}
    canonical_chunks: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank_pos, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank_pos))
            if cid not in canonical_chunks:
                canonical_chunks[cid] = chunk.model_copy(deep=True)

    # Sort chunks by fused score in descending order
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: list[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(sorted_items, start=1):
        chunk = canonical_chunks[cid]
        chunk.score = float(score)
        chunk.rank = rank
        chunk.source = "rrf"
        fused_results.append(chunk)

    return fused_results
