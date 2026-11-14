"""Document reranker operating over retrieved candidate chunks."""

from __future__ import annotations

from policy_rag.providers.base import RerankProvider
from policy_rag.retrieval.clause_graph import logical_doc_id_of
from policy_rag.schemas.retrieval import RetrievedChunk


class DocumentReranker:
    """Reranks candidate chunks using Cross-Encoder with gated 1-Hop graph slot injection."""

    def __init__(self, provider: RerankProvider) -> None:
        self.provider = provider

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int = 5,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        limit = top_k if top_k is not None else top_n
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

        # Gated Graph Reranking: score all direct & graph candidates together
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
            anchor_count = min(2, len(direct_scored), limit)

            for chunk, _ in direct_scored[:anchor_count]:
                selected.append(chunk)
                covered_docs.add(logical_doc_id_of(chunk))

            rem_direct = direct_scored[anchor_count:]

            # 2. Allow up to 1 additional high-scoring novel direct doc at slot 3 to protect Recall@3
            novel_direct_idx: int | None = None
            if len(selected) < limit:
                for idx, (chunk, _) in enumerate(rem_direct):
                    doc_id = logical_doc_id_of(chunk)
                    if doc_id not in covered_docs:
                        novel_direct_idx = idx
                        selected.append(chunk)
                        covered_docs.add(doc_id)
                        break

            # 3. Inject complementary 1-Hop graph chunks (distinct docs) into remaining Top-5 slots
            max_graph_inject = min(2, len(graph_scored), max(0, limit - len(selected)))
            injected_graph = 0
            for chunk, _ in graph_scored:
                if injected_graph >= max_graph_inject or len(selected) >= limit:
                    break
                doc_id = logical_doc_id_of(chunk)
                if doc_id not in covered_docs:
                    selected.append(chunk)
                    covered_docs.add(doc_id)
                    injected_graph += 1

            # 4. Fill remaining budget up to limit with leftover direct candidates
            for idx, (chunk, _) in enumerate(rem_direct):
                if len(selected) >= limit:
                    break
                if idx == novel_direct_idx:
                    continue
                selected.append(chunk)

        for new_rank, chunk in enumerate(selected[:limit], start=1):
            chunk.rank = new_rank

        return selected[:limit]
