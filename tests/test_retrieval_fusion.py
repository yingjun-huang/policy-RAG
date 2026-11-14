"""Unit tests for BM25, Vector search, and Reciprocal Rank Fusion (RRF)."""

import pytest

from policy_rag.providers.fake import HashEmbeddingProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.fusion import reciprocal_rank_fusion
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
    VectorRetriever,
)
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk


def _make_chunk(cid: str, title: str, section: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=f"doc_{cid}",
        chunk_id=cid,
        title=title,
        section=section,
        content=content,
        score=0.0,
        source="test",
    )


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        _make_chunk("c1", "退货政策", "7天无理由", "普通商品自签收后7天内可申请无理由退货。"),
        _make_chunk("c2", "退货政策", "定制商品", "印字刻字等特殊定制商品不适用7天无理由。"),
        _make_chunk("c3", "配送政策", "包邮门槛", "实付满59元包邮，新疆西藏偏远地区需额外加收运费。"),
    ]


def test_bm25_retriever_exact_match(sample_chunks):
    retriever = BM25Retriever(sample_chunks)

    # 1. Exact keyword match on "定制"
    results = retriever.search("定制可以退吗", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"
    assert results[0].source == "bm25"
    assert results[0].score > 0

    # 2. Match on "59元"
    results_shipping = retriever.search("满59包邮吗", top_k=1)
    assert len(results_shipping) == 1
    assert results_shipping[0].chunk_id == "c3"


@pytest.mark.asyncio
async def test_vector_retriever(sample_chunks):
    embedder = HashEmbeddingProvider(dim=64)
    store = InMemoryVectorStore()

    texts = [f"[{c.title} - {c.section}] {c.content}" for c in sample_chunks]
    embeddings = await embedder.embed(texts)
    records = [VectorRecord(chunk=c, embedding=emb) for c, emb in zip(sample_chunks, embeddings)]
    store.upsert(records)

    retriever = VectorRetriever(store, embedder)
    results = await retriever.search("退货条件", top_k=2)

    assert len(results) == 2
    assert results[0].source == "vector"
    assert results[0].rank == 1


def test_rrf_fusion_merges_and_ranks(sample_chunks):
    list_a = [sample_chunks[0], sample_chunks[1]]  # c1 rank 1, c2 rank 2
    list_b = [sample_chunks[1], sample_chunks[2]]  # c2 rank 1, c3 rank 2

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    # c2 appeared in both lists (rank 2 in A, rank 1 in B), so its RRF score should be highest!
    # c2 score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.01613 + 0.01639 = 0.0325
    # c1 score: 1/(60+1) ~ 0.01639
    assert len(fused) == 3
    assert fused[0].chunk_id == "c2"
    assert fused[0].rank == 1
    assert fused[0].source == "rrf"


@pytest.mark.asyncio
async def test_hybrid_retriever_end_to_end(sample_chunks):
    bm25 = BM25Retriever(sample_chunks)
    embedder = HashEmbeddingProvider(dim=64)
    store = InMemoryVectorStore()

    texts = [c.content for c in sample_chunks]
    embeddings = await embedder.embed(texts)
    store.upsert([VectorRecord(chunk=c, embedding=e) for c, e in zip(sample_chunks, embeddings)])

    vector = VectorRetriever(store, embedder)
    hybrid = HybridRetriever(bm25, vector, rrf_k=60)

    results = await hybrid.retrieve("偏远地区运费", top_k=2)
    assert len(results) <= 2
    assert any(c.chunk_id == "c3" for c in results)


@pytest.mark.asyncio
async def test_qdrant_vector_store_and_scalar_pre_filter():
    embedder = HashEmbeddingProvider(dim=64)
    qdrant_store = QdrantVectorStore(collection="test_policy_qdrant", dim=64)

    c_active = RetrievedChunk(
        document_id="jd_ship_01_freight_standard_11111111",
        chunk_id="jd_ship_01_freight_standard_11111111_00",
        title="自营运费标准",
        section="基础运费",
        content="普通用户实付商品金额满59元免基础运费。",
        metadata={
            "logical_doc_id": "jd_ship_01_freight_standard",
            "doc_group": "shipping",
            "document_status": "current",
            "effective_from": "2023-08-23",
        },
    )
    c_archived = RetrievedChunk(
        document_id="jd_ver_01_shipping_2023_legacy_22222222",
        chunk_id="jd_ver_01_shipping_2023_legacy_22222222_00",
        title="自营运费旧版",
        section="旧版门槛",
        content="普通用户实付商品金额满99元免基础运费。",
        metadata={
            "logical_doc_id": "jd_ver_01_shipping_2023_legacy",
            "doc_group": "legacy_archive",
            "document_status": "archived",
            "effective_from": "2016-04-01",
            "effective_to": "2023-08-23",
            "effective_to_exclusive": True,
        },
    )
    c_cat = RetrievedChunk(
        document_id="jd_cat_01_mobile_phones_33333333",
        chunk_id="jd_cat_01_mobile_phones_33333333_00",
        title="手机退货规则",
        section="激活限制",
        content="手机激活后不支持七天无理由退货。",
        metadata={
            "logical_doc_id": "jd_cat_01_mobile_phones",
            "doc_group": "category_rules",
            "document_status": "current",
            "effective_from": "2024-01-01",
        },
    )

    chunks = [c_active, c_archived, c_cat]
    embeddings = await embedder.embed([c.content for c in chunks])
    qdrant_store.upsert([VectorRecord(chunk=c, embedding=e) for c, e in zip(chunks, embeddings)])

    assert len(qdrant_store) == 3
    retriever = VectorRetriever(qdrant_store, embedder)

    # 1. Default active filter excludes archived legacy document via Qdrant scalar pre-filter
    active_filter = MetadataFilter(as_of="2024-06-01", include_archived=False)
    res_active = await retriever.retrieve("免基础运费", top_k=5, metadata_filter=active_filter)
    returned_ids = {c.chunk_id for c in res_active}
    assert c_active.chunk_id in returned_ids
    assert c_archived.chunk_id not in returned_ids

    # 2. Category doc_group filter restricts search to category_rules only
    cat_filter = MetadataFilter(as_of="2024-06-01", include_archived=False, doc_groups=["category_rules"])
    res_cat = await retriever.retrieve("手机退货", top_k=5, metadata_filter=cat_filter)
    assert len(res_cat) == 1
    assert res_cat[0].chunk_id == c_cat.chunk_id

    # 3. Historical contrast filter unlocks archived document
    contrast_filter = MetadataFilter(as_of=None, include_archived=True)
    res_all = await retriever.retrieve("免基础运费", top_k=5, metadata_filter=contrast_filter)
    assert c_archived.chunk_id in {c.chunk_id for c in res_all}


def test_lex_specialis_priority_and_lost_in_the_middle_formatting():
    from policy_rag.agents.loop import _format_evidence_for_prompt
    from policy_rag.retrieval.filters import expand_parent_contexts, resolve_authority_priority

    p3 = RetrievedChunk(
        document_id="jd_rule_01_doc_00",
        chunk_id="c_p3",
        title="通用售后规则",
        section="通则",
        content="一般商品适用7天无理由。",
        source="clause_graph",
        metadata={
            "logical_doc_id": "jd_rule_01_general",
            "doc_group": "general_rules",
            "graph_expanded": True,
            "graph_edge": "INHERITS_GENERAL",
        },
    )
    p2 = RetrievedChunk(
        document_id="jd_cat_01_doc_00",
        chunk_id="c_p2",
        title="手机品类退货特例",
        section="激活限制",
        content="手机激活后不支持7天无理由。",
        metadata={"logical_doc_id": "jd_cat_01_mobile_phones", "doc_group": "category_rules"},
    )
    p1 = RetrievedChunk(
        document_id="jd_srv_02_doc_00",
        chunk_id="c_p1",
        title="价保专项服务",
        section="价保规则",
        content="价保期间支持一键退差价。",
        metadata={"logical_doc_id": "jd_srv_02_price_protection", "doc_group": "service_rules"},
    )
    p2_extra = RetrievedChunk(
        document_id="jd_ship_01_doc_00",
        chunk_id="c_p2_ship",
        title="自营运费标准",
        section="偏远运费",
        content="偏远地区需另计基础运费。",
        metadata={"logical_doc_id": "jd_ship_01_freight", "doc_group": "shipping"},
    )

    assert resolve_authority_priority(p1)[0] == 1
    assert resolve_authority_priority(p2)[0] == 2
    assert resolve_authority_priority(p3)[0] == 3

    # Ranked by authority tier -> [p1, p2, p2_extra, p3]
    # Lost-in-the-middle puts 0 at start, 1 at end, 2 at index 1, 3 at index -2 -> [p1, p2_extra, p3, p2]
    ordered = expand_parent_contexts([p3, p2, p1, p2_extra])
    assert [c.chunk_id for c in ordered] == ["c_p1", "c_p2_ship", "c_p3", "c_p2"]

    formatted = _format_evidence_for_prompt(ordered)
    assert "[效力优先级: P1-专项服务与特定版本规则（最高优先适用）]" in formatted
    assert "[效力优先级: P2-品类特例与专项运费细则（优先于通用总则）]" in formatted
    assert "[效力优先级: P3-平台通用基础总则（无特例冲突时适用）]" in formatted
    assert "[召回来源: 图谱1-Hop关联补全 (INHERITS_GENERAL)]" in formatted

