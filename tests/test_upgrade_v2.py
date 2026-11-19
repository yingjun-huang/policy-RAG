"""Comprehensive tests for RAG knowledge base & pipeline upgrade (v2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.build_dataset import build_datasets, load_policy_texts, normalize_benchmark_entry
from policy_rag.agents.graph import build_graph
from policy_rag.agents.loop import _format_evidence_for_prompt
from policy_rag.agents.nodes import faithfulness_node
from policy_rag.config import Settings
from policy_rag.indexing.chunking import chunk_markdown, parse_frontmatter, parse_version_banner
from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.providers.fake import FakeLLMProvider, HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.filters import expand_parent_contexts, infer_retrieval_filter, matches_metadata_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import InMemoryVectorStore, VectorRecord, VectorRetriever
from policy_rag.schemas.chat import ChatMessage, ChatRequest
from policy_rag.schemas.retrieval import (
    Citation,
    ExitReason,
    GradeStatus,
    MetadataFilter,
    ResponseAction,
    RetrievedChunk,
)
from policy_rag.service import PolicyContainer

ROOT = Path(__file__).resolve().parent.parent


def test_yaml_frontmatter_stripped_and_metadata_normalized():
    kb_dir = ROOT / "knowledge_jd"
    md_files = sorted(kb_dir.glob("*.md"))
    assert len(md_files) == 32
    for p in md_files:
        assert not p.read_text(encoding="utf-8").lstrip().startswith("---"), f"{p.name} must not have YAML"

    chunks = load_and_chunk_knowledge(kb_dir)
    assert len(chunks) >= 90

    # Zero chunks should contain raw YAML delimiters, stub sections, or missing version metadata
    for c in chunks:
        assert not c.content.lstrip().startswith("---")
        assert "document_status:" not in c.content
        assert c.section != "概述"
        assert c.parent_id is not None
        assert "parent_content" in c.metadata
        assert c.metadata.get("effective_from")
        assert c.metadata.get("doc_group") in {
            "category_rules",
            "shipping",
            "service_rules",
            "general_rules",
            "legacy_archive",
        }

    archived_ids = {
        c.metadata.get("logical_doc_id")
        for c in chunks
        if c.metadata.get("document_status") == "archived"
    }
    assert archived_ids == {"jd_ver_01_shipping_2023_legacy", "jd_ver_02_fresh_2024_legacy"}


def test_scheme_a_version_banner_extraction_and_prompt_visibility():
    archived_md = """# 自营配送旧版本：2016–2023（归档）

政策版本：2016-04-v1（已归档失效） | 生效期：2016-04-01 至 2023-08-23 前 | 适用事件：订单支付时间

## 旧版门槛

普通用户实付商品金额满99元免基础运费。
"""
    banner = parse_version_banner("政策版本：2016-04-v1（已归档失效） | 生效期：2016-04-01 至 2023-08-23 前 | 适用事件：订单支付时间")
    assert banner["version"] == "2016-04-v1"
    assert banner["effective_from"] == "2016-04-01"
    assert banner["effective_to"] == "2023-08-23"
    assert banner["document_status"] == "archived"
    assert banner["applicability_event"] == "order_paid_at"

    chunks = chunk_markdown(archived_md, document_id="jd_ver_01_x", doc_slug="jd_ver_01_shipping_2023_legacy")
    assert len(chunks) == 1
    assert chunks[0].section == "旧版门槛"
    assert chunks[0].metadata["doc_group"] == "legacy_archive"
    assert "已归档失效" in chunks[0].metadata["parent_content"]

    prompt_txt = _format_evidence_for_prompt(chunks)
    assert "已归档失效" in prompt_txt
    assert "2023-08-23 前" in prompt_txt


def test_parse_frontmatter_and_table_preservation():
    sample_md = """---
doc_id: jd_test_table
title: 测试表文档
document_status: current
effective_from: 2024-01-01
---
# 测试表文档

## 计算公式与费率表

说明文字在表格前面。

| 品类 | 费率 | 备注 |
|---|---|---|
| 冰箱 | 0.05% | 测试费率1 |
| 电视 | 0.05% | 测试费率2 |
| 洗衣机 | 0.05% | 测试费率3 |
| 空调 | 0.05% | 测试费率4 |
| 微波炉 | 0.10% | 测试费率5 |
| 吸尘器 | 0.10% | 测试费率6 |
| 电饭煲 | 0.10% | 测试费率7 |
| 电磁炉 | 0.10% | 测试费率8 |
"""
    fm, body = parse_frontmatter(sample_md)
    assert fm["doc_id"] == "jd_test_table"
    assert not body.lstrip().startswith("---")

    # Force small chunk size so table splits across multiple sub-chunks cleanly
    chunks = chunk_markdown(sample_md, document_id="jd_test_table", max_chunk_chars=180)
    assert len(chunks) >= 2
    for c in chunks:
        lines = [ln for ln in c.content.splitlines() if ln.strip().startswith("|")]
        if lines:
            # Every chunk holding table data rows must retain its table header & separator row
            assert any("| 品类 |" in ln for ln in lines)
            assert any("|---|" in ln for ln in lines)


def test_parent_expansion_reunites_appliances_table():
    chunks = load_and_chunk_knowledge(ROOT / "knowledge_jd")
    appliances_chunks = [
        c for c in chunks if c.metadata.get("logical_doc_id") == "jd_cat_03_appliances_large"
    ]
    assert len(appliances_chunks) >= 2

    expanded = expand_parent_contexts(appliances_chunks[:3], enabled=True, max_parent_chars=1500)
    assert len(expanded) == 1
    assert expanded[0].chunk_id == appliances_chunks[0].chunk_id
    assert "0.05%" in expanded[0].content
    assert "日折旧率" in expanded[0].content


def test_infer_retrieval_filter_version_isolation_and_correction():
    # Normal depreciation query ("折旧") must NOT trigger archived versions
    f_dep = infer_retrieval_filter("冰箱发票4000元签收第30天折旧多少钱？")
    assert f_dep.include_archived is False

    # Normal active shipping query must NOT trigger archived versions
    f_ship = infer_retrieval_filter("普通用户上海70元3kg自营中小件运费是多少？", policy_context={"as_of": "2024-06-01"})
    # A supplied date selects the effective version even when archives are eligible.
    old = RetrievedChunk(document_id="old", chunk_id="old", title="旧版", content="旧门槛",
                         metadata={"document_status":"archived","effective_to":"2024-01-01"})
    current = RetrievedChunk(document_id="current", chunk_id="current", title="新版", content="新门槛",
                             metadata={"document_status":"current","effective_from":"2024-01-01"})
    assert not matches_metadata_filter(old, f_ship)
    assert matches_metadata_filter(current, f_ship)
    assert f_ship.as_of == "2024-06-01"

    # Explicit version contrast unlocks archived rules
    f_vc = infer_retrieval_filter("2023年满99元免邮门槛和2024年新版59元门槛区别是什么？")
    assert f_vc.include_archived is True

    # User correction changing 2023 -> 2024 must isolate archived 2023 doc again
    f_corr = infer_retrieval_filter(
        "不对我看错了，是2024年3月付款实付65元，运费多少？",
        history=[{"role": "user", "content": "2023年7月实付50元运费是多少？"}],
    )
    assert f_corr.include_archived is False


@pytest.mark.asyncio
async def test_shared_prefusion_metadata_filter_blocks_archived_in_bm25_and_vector():
    current_chunk = RetrievedChunk(
        document_id="jd_ship_01_aaaabbbb",
        chunk_id="jd_ship_01_00",
        title="自营普通配送计费",
        content="普通用户实付商品金额满59元免基础运费，未满59元基础运费6元。",
        section="基础运费",
        metadata={
            "logical_doc_id": "jd_ship_01_freight_standard",
            "document_status": "current",
            "effective_from": "2024-01-01",
            "effective_to": None,
        },
    )
    archived_chunk = RetrievedChunk(
        document_id="jd_ver_01_ccccdddd",
        chunk_id="jd_ver_01_00",
        title="自营配送旧版本：2016–2023",
        content="普通用户实付商品金额满99元免基础运费，未满99元基础费6元。",
        section="旧版门槛",
        metadata={
            "logical_doc_id": "jd_ver_01_shipping_2023_legacy",
            "document_status": "archived",
            "effective_from": "2016-04-01",
            "effective_to": "2023-08-23",
        },
    )
    corpus = [current_chunk, archived_chunk]
    bm25 = BM25Retriever(corpus)
    embedder = HashEmbeddingProvider(dim=64)
    vec_store = InMemoryVectorStore()
    embs = await embedder.embed([c.content for c in corpus])
    vec_store.upsert([VectorRecord(chunk=c, embedding=e) for c, e in zip(corpus, embs)])
    vector = VectorRetriever(vec_store, embedder)
    hybrid = HybridRetriever(bm25, vector)

    active_filter = MetadataFilter(as_of="2024-06-01", include_archived=False)
    assert [c.chunk_id for c in bm25.search("满99元 基础运费", top_k=5, metadata_filter=active_filter)] == ["jd_ship_01_00"]
    assert [c.chunk_id for c in await vector.search("满99元 基础运费", top_k=5, metadata_filter=active_filter)] == ["jd_ship_01_00"]
    assert [c.chunk_id for c in await hybrid.retrieve("满99元 基础运费", top_k=5, metadata_filter=active_filter)] == ["jd_ship_01_00"]

    contrast_filter = MetadataFilter(as_of=None, include_archived=True)
    contrast_ids = {c.chunk_id for c in await hybrid.retrieve("99元 59元 基础运费", top_k=5, metadata_filter=contrast_filter)}
    assert contrast_ids == {"jd_ship_01_00", "jd_ver_01_00"}


@pytest.mark.asyncio
async def test_empty_evidence_abstains_without_fake_fallback():
    empty_bm25 = BM25Retriever([])
    empty_store = InMemoryVectorStore()
    embedder = HashEmbeddingProvider(dim=64)
    hybrid = HybridRetriever(empty_bm25, VectorRetriever(empty_store, embedder))
    reranker = DocumentReranker(LexicalRerankProvider())
    llm = FakeLLMProvider(default_response='{"strategy": "single", "rationale": "test"}')

    graph = build_graph(retriever=hybrid, reranker=reranker, llm=llm)
    res = await graph.ainvoke({"user_query": "京东白条提前还款违约金怎么算？"})

    assert res["evidence"] == []
    assert res["citations"] == []
    assert res["action"] == ResponseAction.ABSTAIN
    assert res["exit_reason"] == ExitReason.NO_EVIDENCE
    assert all(c.chunk_id != "return_policy_00" for c in res["evidence"])


@pytest.mark.asyncio
async def test_faithfulness_string_false_triggers_self_correction():
    chunk = RetrievedChunk(
        document_id="doc_1",
        chunk_id="doc_1_00",
        title="破损包赔",
        content="密封液体实际渗漏比例>30%全额退款，≤30%按比例退款。",
        section="赔付标准",
    )
    canned = {
        "审核 JSON": '{"supported": "false", "hallucinated_spans": ["一律退双倍"], "rationale": "条款仅全额退款"}',
        "修正后的 JSON": '{"action": "answer", "answer": "渗漏超过30%可全额商品退款。", "citation_chunk_ids": ["doc_1_00"]}',
    }
    llm = FakeLLMProvider(canned=canned)
    state = {
        "user_query": "密封液体渗漏40%赔多少？",
        "evidence": [chunk],
        "final_answer": "渗漏40%一律退双倍商品金额。",
        "citations": [Citation(document_id="doc_1", title="破损包赔", chunk_id="doc_1_00", section="赔付标准")],
        "refused": False,
    }
    updates = await faithfulness_node(state, llm=llm)
    assert updates["final_answer"] == "渗漏超过30%可全额商品退款。"
    assert updates["citations"][0].chunk_id == "doc_1_00"


@pytest.mark.asyncio
async def test_multiturn_session_history_and_policy_context_api():
    settings = Settings(app_env="testing", knowledge_dir=str(ROOT / "knowledge_jd"))
    container = PolicyContainer(settings=settings)

    r1 = await container.qa_service.answer(
        ChatRequest(
            message="手机收到好几天了，现在还能退吗？",
            session_id="u1_phone",
            policy_context={"as_of": "2024-06-01T12:00:00+08:00"},
        )
    )
    assert r1.refused is False
    assert r1.action == ResponseAction.ANSWER
    assert len(container.qa_service.sessions["u1_phone"]) == 2

    # Turn 2 uses session memory so short answer resolves to mobile phone rules
    r2 = await container.qa_service.answer(
        ChatRequest(
            message="拆封没激活，签收第5天。",
            session_id="u1_phone",
            policy_context={"as_of": "2024-06-01T12:00:00+08:00"},
        )
    )
    ev_docs = {c.metadata.get("logical_doc_id") for c in r2.evidence}
    assert "jd_cat_01_mobile_phones" in ev_docs

    # Explicit ChatRequest.history contract also works cleanly
    r3 = await container.qa_service.answer(
        ChatRequest(
            message="更正一下，实际我买的是刻了字的高级定制品且已投入生产。",
            history=[ChatMessage(role="user", content="普通礼盒可以7天无理由退货吗？")],
            policy_context={"as_of": "2024-06-01"},
        )
    )
    assert "jd_cat_11_custom_products" in {c.metadata.get("logical_doc_id") for c in r3.evidence}


def test_eval_datasets_present_and_well_formed():
    bench = [
        json.loads(ln)
        for ln in (ROOT / "eval" / "dataset" / "jd_eval_benchmark.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    guard = [
        json.loads(ln)
        for ln in (ROOT / "eval" / "dataset" / "jd_eval_guardrail_and_correction.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(bench) >= 100
    assert len(guard) >= 15

    shards_dir = ROOT / "eval" / "dataset" / "shards"
    assert shards_dir.is_dir()
    report = build_datasets(
        ROOT,
        include_implicit=any(row.get("category") == "implicit_temporal" for row in bench),
        check_only=True,
    )
    assert report["status"] == "PASS", report["errors"]


def test_build_dataset_auto_resolves_concise_shard_entries():
    texts = load_policy_texts(ROOT / "knowledge_jd")
    concise_item = {
        "id": "jd_case_test_concise",
        "query": "今天收到一单生鲜水果，请问签收后多久内要提交照片申请优鲜赔？",
        "gold_evidence_refs": [
            {
                "doc_id": "jd_srv_01_fresh_instant_pay",
                "section": "反馈窗口",
                "quote": "活鲜签收后6小时内、其他适用果蔬肉蛋签收后48小时内提交面单与坏损凭证，均包含恰好截止小时；必须是实际提交时间，不能以仅拍照时间代替提交。具体退款比例按对应品类，未报损或未核实不能预先保证退款。",
            }
        ],
        "gold_answer": "水果属于适用果蔬，须在签收后48小时内实际提交面单与坏损凭证。",
    }
    normalized = normalize_benchmark_entry(concise_item, texts, expected_category="single_hop")
    assert normalized["gold_doc_ids"] == ["jd_srv_01_fresh_instant_pay"]
    assert normalized["relevant_doc_ids"] == ["jd_srv_01_fresh_instant_pay"]
    assert len(normalized["gold_evidence_spans"]) == 1
    ref = normalized["gold_evidence_refs"][0]
    assert ref["line_start"] == ref["line_end"] == 11
    assert texts["jd_srv_01_fresh_instant_pay"][ref["char_start"] : ref["char_end"]] == ref["quote"]
    assert len(ref["document_sha256"]) == 64


def test_generic_llm_settings_override_deepseek():
    hub = Settings(
        llm_api_key="sk-aihubmix-test",
        llm_base_url="https://api.inferera.com/v1",
        llm_model="ernie-5.5-flash-preview-free",
        deepseek_api_key="sk-deepseek-legacy",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
    )
    assert hub.llm_configured is True
    assert hub.active_llm_api_key == "sk-aihubmix-test"
    assert hub.active_llm_base_url == "https://api.inferera.com/v1"
    assert hub.active_llm_model == "ernie-5.5-flash-preview-free"

    fallback = Settings(
        llm_api_key="",
        llm_base_url="",
        llm_model="",
        deepseek_api_key="sk-deepseek-legacy",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
    )
    assert fallback.llm_configured is True
    assert fallback.active_llm_api_key == "sk-deepseek-legacy"
    assert fallback.active_llm_base_url == "https://api.deepseek.com"
    assert fallback.active_llm_model == "deepseek-chat"

    offline = Settings(
        llm_api_key="",
        deepseek_api_key="",
    )
    assert offline.llm_configured is False
