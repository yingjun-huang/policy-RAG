"""Document reranker operating over retrieved candidate chunks with CE-threshold graph gating."""

from __future__ import annotations

from policy_rag.providers.base import RerankProvider
from policy_rag.retrieval.clause_graph import logical_doc_id_of
from policy_rag.schemas.retrieval import RetrievedChunk


class DocumentReranker:
    """Reranks candidate chunks using Cross-Encoder with CE-score gated 1-Hop graph slot injection."""

    def __init__(
        self,
        provider: RerankProvider,
        *,
        graph_ce_threshold: float | None = 0.5,
    ) -> None:
        self.provider = provider
        self.graph_ce_threshold = graph_ce_threshold

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int = 5,
        top_k: int | None = None,
        graph_ce_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        limit = top_k if top_k is not None else top_n
        threshold = self.graph_ce_threshold if graph_ce_threshold is None else graph_ce_threshold
        doc_texts = [f"[{c.title} - {c.section or '条款'}] {c.content}" for c in chunks]

        has_graph = any(bool((c.metadata or {}).get("graph_expanded")) for c in chunks)
        if not has_graph:
            ranked = await self.provider.rerank(query, doc_texts, top_n=min(limit, len(chunks)))
            results: list[RetrievedChunk] = []
            for new_rank, (original_idx, score) in enumerate(ranked, start=1):
                chunk = chunks[original_idx].model_copy(deep=True)
                chunk.score = float(score)
                chunk.rank = new_rank
                chunk.source = "rerank"
                results.append(chunk)
            return results

        # Score all direct & graph candidates together with Cross-Encoder
        all_ranked = await self.provider.rerank(query, doc_texts, top_n=len(chunks))

        direct_scored: list[tuple[RetrievedChunk, float]] = []
        graph_scored: list[tuple[RetrievedChunk, float]] = []
        for original_idx, score in all_ranked:
            chunk = chunks[original_idx].model_copy(deep=True)
            chunk.score = float(score)
            chunk.source = "rerank"
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["rerank_score"] = float(score)
            if chunk.metadata.get("graph_expanded"):
                graph_scored.append((chunk, float(score)))
            else:
                direct_scored.append((chunk, float(score)))

        if not direct_scored:
            selected = [c for c, _ in graph_scored[:limit]]
        else:
            # 1. Preserve Top-1 ~ Top-2 strictly from direct recall to protect Precision@1
            selected: list[RetrievedChunk] = []
            covered_docs: set[str] = set()
            selected_ids: set[str] = set()
            anchor_count = min(2, len(direct_scored), limit)

            for chunk, _ in direct_scored[:anchor_count]:
                selected.append(chunk)
                covered_docs.add(logical_doc_id_of(chunk))
                selected_ids.add(chunk.chunk_id)

            rem_direct = direct_scored[anchor_count:]

            # 2. Allow up to 1 additional novel direct doc at slot 3 to protect Recall@3
            if len(selected) < limit:
                for chunk, _ in rem_direct:
                    doc_id = logical_doc_id_of(chunk)
                    if doc_id not in covered_docs:
                        selected.append(chunk)
                        covered_docs.add(doc_id)
                        selected_ids.add(chunk.chunk_id)
                        break

            # 3. CE-Gated injection: only graph candidates with score >= threshold get reserved complementary slots
            eligible_graph = [
                (c, s)
                for c, s in graph_scored
                if threshold is None or s >= threshold
            ]
            max_graph_inject = min(2, len(eligible_graph), max(0, limit - len(selected)))
            injected_graph = 0
            for chunk, _ in eligible_graph:
                if injected_graph >= max_graph_inject or len(selected) >= limit:
                    break
                doc_id = logical_doc_id_of(chunk)
                if doc_id not in covered_docs:
                    selected.append(chunk)
                    covered_docs.add(doc_id)
                    selected_ids.add(chunk.chunk_id)
                    injected_graph += 1

            # 4. Remaining pool: unselected direct + below-threshold graph candidates compete strictly by CE score
            leftover: list[tuple[RetrievedChunk, float]] = [
                (c, s)
                for c, s in (rem_direct + graph_scored)
                if c.chunk_id not in selected_ids
            ]
            leftover.sort(key=lambda item: item[1], reverse=True)
            for chunk, _ in leftover:
                if len(selected) >= limit:
                    break
                selected.append(chunk)
                selected_ids.add(chunk.chunk_id)

        for new_rank, chunk in enumerate(selected[:limit], start=1):
            chunk.rank = new_rank

        return selected[:limit]
