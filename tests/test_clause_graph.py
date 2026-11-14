"""Unit tests for NetworkX ClauseDependencyGraph and Gated Graph Reranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.providers.fake import HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.clause_graph import (
    EDGE_INHERITS_GENERAL,
    EDGE_LINKS_COMPLEMENT,
    EDGE_REQUIRES_DISPUTE_PROOF,
    ClauseDependencyGraph,
    logical_doc_id_of,
)
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import InMemoryVectorStore, VectorRecord, VectorRetriever
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk


def _chunk(
    doc_id: str,
    chunk_id: str,
    title: str,
    content: str,
    *,
    graph_expanded: bool = False,
    doc_group: str = "category_rules",
    status: str = "current",
) -> RetrievedChunk:
    meta = {
        "logical_doc_id": doc_id,
        "doc_group": doc_group,
        "document_status": status,
    }
    if graph_expanded:
        meta["graph_expanded"] = True
        meta["graph_edge"] = EDGE_INHERITS_GENERAL
    return RetrievedChunk(
        document_id=f"{doc_id}_deadbeef",
        chunk_id=chunk_id,
        title=title,
        section="条款",
        content=content,
        metadata=meta,
    )


def test_clause_graph_builds_on_real_jd_corpus():
    knowledge_dir = Path(__file__).resolve().parent.parent / "knowledge_jd"
    chunks = load_and_chunk_knowledge(knowledge_dir)
    cg = ClauseDependencyGraph(chunks)

    assert cg.node_count == 32
    assert cg.edge_count >= 20
    assert cg.graph.has_edge("jd_cat_07_fashion_apparel", "jd_rule_01_seven_days_general")
    assert cg.graph["jd_cat_07_fashion_apparel"]["jd_rule_01_seven_days_general"]["edge_type"] == EDGE_INHERITS_GENERAL
    assert cg.graph.has_edge("jd_cat_11_custom_products", "jd_rule_02_dispute_resolution")
    assert cg.graph["jd_cat_11_custom_products"]["jd_rule_02_dispute_resolution"]["edge_type"] == EDGE_REQUIRES_DISPUTE_PROOF
    assert cg.graph.has_edge("jd_cat_06_fresh_seafood", "jd_srv_01_fresh_instant_pay")
    assert cg.graph["jd_cat_06_fresh_seafood"]["jd_srv_01_fresh_instant_pay"]["edge_type"] == EDGE_LINKS_COMPLEMENT


def test_clause_graph_expands_missing_general_and_dispute_docs():
    c_apparel = _chunk("jd_cat_07_fashion_apparel", "c1", "服饰退换规范", "吊牌防伪扣完好可退")
    c_general = _chunk(
        "jd_rule_01_seven_days_general",
        "c2",
        "七天无理由退货总则",
        "自签收次日零时起算满168小时为7天",
        doc_group="general_rules",
    )
    c_custom = _chunk("jd_cat_11_custom_products", "c3", "定制品规范", "定制窗帘误差超2cm可修改")
    c_dispute = _chunk(
        "jd_rule_02_dispute_resolution",
        "c4",
        "交易纠纷举证总则",
        "双方争议按举证与第三方检测认定",
        doc_group="general_rules",
    )

    cg = ClauseDependencyGraph([c_apparel, c_general, c_custom, c_dispute])

    # Case 1: apparel return -> expands jd_rule_01_seven_days_general
    expanded_apparel = cg.expand_candidates("衣服不想要了，现在还能退吗？", [c_apparel])
    stems_apparel = [logical_doc_id_of(c) for c in expanded_apparel]
    assert "jd_rule_01_seven_days_general" in stems_apparel
    added = [c for c in expanded_apparel if c.metadata.get("graph_expanded")]
    assert len(added) == 1
    assert added[0].metadata["graph_edge"] == EDGE_INHERITS_GENERAL

    # Case 2: custom curtain size dispute -> expands jd_rule_02_dispute_resolution
    expanded_custom = cg.expand_candidates("定制窗帘看着长了怀疑尺寸不对，能退货退款吗？", [c_custom])
    stems_custom = [logical_doc_id_of(c) for c in expanded_custom]
    assert "jd_rule_02_dispute_resolution" in stems_custom
    assert "jd_rule_01_seven_days_general" in stems_custom


def test_clause_graph_respects_metadata_filter():
    c_apparel = _chunk("jd_cat_07_fashion_apparel", "c1", "服饰规范", "衣服可退换", doc_group="category_rules")
    c_general = _chunk(
        "jd_rule_01_seven_days_general",
        "c2",
        "总则",
        "七天起算规则",
        doc_group="general_rules",
    )
    cg = ClauseDependencyGraph([c_apparel, c_general])

    # Restrict doc_groups strictly to category_rules -> jd_rule_01 (general_rules) must not be expanded
    restricted = MetadataFilter(doc_groups=["category_rules"], include_archived=False)
    expanded = cg.expand_candidates("衣服能退吗", [c_apparel], metadata_filter=restricted)
    assert [logical_doc_id_of(c) for c in expanded] == ["jd_cat_07_fashion_apparel"]


@pytest.mark.asyncio
async def test_gated_graph_reranking_preserves_top1_and_injects_graph_slot():
    reranker = DocumentReranker(LexicalRerankProvider())

    # c1 and c2 are strong direct hits; c_unrelated is a weak direct hit; c_graph is expanded doc
    chunks = [
        _chunk("jd_cat_07_fashion_apparel", "d1", "服饰规范", "衣服不想要了试穿吊牌完好可退"),
        _chunk("jd_cat_07_fashion_apparel", "d2", "服饰规范", "衣服不想要了洗涤不退"),
        _chunk("jd_cat_08_beauty_cosmetics", "d3", "美妆规范", "美妆衣服退货说明"),
        _chunk("jd_cat_09_jewelry_watches", "d4", "珠宝规范", "珠宝退货说明"),
        _chunk(
            "jd_rule_01_seven_days_general",
            "g1",
            "七天无理由总则",
            "签收次日零时起算168小时",
            graph_expanded=True,
            doc_group="general_rules",
        ),
    ]

    reranked = await reranker.rerank("衣服不想要了现在还能退吗", chunks, top_n=5)
    assert len(reranked) == 5
    # Rank 1 and Rank 2 must strictly come from direct hits (not graph_expanded)
    assert not reranked[0].metadata.get("graph_expanded")
    assert not reranked[1].metadata.get("graph_expanded")
    assert logical_doc_id_of(reranked[0]) == "jd_cat_07_fashion_apparel"
    # Graph expanded chunk must be injected inside Top-5 complementary slots
    assert any(logical_doc_id_of(c) == "jd_rule_01_seven_days_general" for c in reranked)


@pytest.mark.asyncio
async def test_hybrid_retriever_toggle_graph_expansion():
    c1 = _chunk("jd_cat_07_fashion_apparel", "c1", "服饰", "衣服吊牌完好可退")
    c2 = _chunk("jd_rule_01_seven_days_general", "c2", "总则", "签收次日起168小时", doc_group="general_rules")
    chunks = [c1, c2]

    bm25 = BM25Retriever(chunks)
    embedder = HashEmbeddingProvider(dim=64)
    store = InMemoryVectorStore()
    embs = await embedder.embed([c.content for c in chunks])
    store.upsert([VectorRecord(chunk=c, embedding=e) for c, e in zip(chunks, embs)])
    vec = VectorRetriever(store, embedder)
    cg = ClauseDependencyGraph(chunks)

    hybrid = HybridRetriever(bm25, vec, clause_graph=cg)

    res_no_graph = await hybrid.retrieve("衣服", top_k=1, enable_graph_expansion=False)
    assert len(res_no_graph) == 1

    res_with_graph = await hybrid.retrieve("衣服不想要了能退吗", top_k=1, enable_graph_expansion=True)
    stems = {logical_doc_id_of(c) for c in res_with_graph}
    assert "jd_rule_01_seven_days_general" in stems
