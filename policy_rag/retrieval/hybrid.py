"""Hybrid retriever fusing lexical BM25 and dense vector search via RRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from policy_rag.retrieval.vector import VectorRetriever
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk

if TYPE_CHECKING:
    from policy_rag.retrieval.clause_graph import ClauseDependencyGraph


class HybridRetriever:
    """Orchestrates dual-route recall (BM25 + Dense Vector) and merges candidates with RRF."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        clause_graph: ClauseDependencyGraph | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.clause_graph = clause_graph

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
        enable_graph_expansion: bool = True,
    ) -> list[RetrievedChunk]:
        # Over-retrieve candidates for fusion (fetch top_k * 2 from each route)
        candidate_k = max(top_k * 2, 10)

        # Vector is async, BM25 is CPU-bound
        bm25_results = self.bm25.search(query, top_k=candidate_k, metadata_filter=metadata_filter)
        vector_results = await self.vector.search(query, top_k=candidate_k, metadata_filter=metadata_filter)

        # Merge via Cormack RRF
        fused = reciprocal_rank_fusion([bm25_results, vector_results], k=self.rrf_k)
        primary = fused[:top_k]

        if self.clause_graph is not None and enable_graph_expansion:
            return self.clause_graph.expand_candidates(
                query,
                primary,
                metadata_filter=metadata_filter,
            )
        return primary
