"""Tests for YAML-driven ClauseDependencyGraph, CE-threshold rerank gating, and edge suggestions."""

from __future__ import annotations

from pathlib import Path

import pytest

from policy_rag.providers.base import RerankProvider
from policy_rag.retrieval.clause_graph import (
    ClauseDependencyGraph,
    load_and_validate_graph_config,
    logical_doc_id_of,
)
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.schemas.retrieval import RetrievedChunk
from scripts.suggest_edges import extract_candidate_edges


def _chunk(
    doc_id: str,
    chunk_id: str,
    title: str,
    content: str,
    *,
    graph_expanded: bool = False,
    doc_group: str = "category_rules",
) -> RetrievedChunk:
    meta = {
        "logical_doc_id": doc_id,
        "doc_group": doc_group,
        "document_status": "current",
    }
    if graph_expanded:
        meta["graph_expanded"] = True
        meta["graph_edge"] = "INHERITS_GENERAL"
    return RetrievedChunk(
        document_id=f"{doc_id}_deadbeef",
        chunk_id=chunk_id,
        title=title,
        section="条款",
        content=content,
        metadata=meta,
    )


class _FixedScoreRerankProvider:
    backend = "fixed-test"
    available = True

    def __init__(self, scores_by_prefix: dict[str, float]) -> None:
        self._scores_by_prefix = scores_by_prefix

    async def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            score = 0.0
            for prefix, val in self._scores_by_prefix.items():
                if prefix in doc:
                    score = val
                    break
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


def test_policy_graph_yaml_loads_and_validates(tmp_path: Path):
    specs = load_and_validate_graph_config()
    assert len(specs) >= 12

    bad_self_loop = tmp_path / "bad_loop.yaml"
    bad_self_loop.write_text(
        "edges:\n  - {src: doc_a, tgt: doc_a, type: INHERITS_GENERAL, priority: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="self-loop"):
        load_and_validate_graph_config(bad_self_loop)

    bad_type = tmp_path / "bad_type.yaml"
    bad_type.write_text(
        "edges:\n  - {src: doc_a, tgt: doc_b, type: UNKNOWN_EDGE, priority: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported type"):
        load_and_validate_graph_config(bad_type)

    bad_both_src = tmp_path / "bad_src.yaml"
    bad_both_src.write_text(
        "edges:\n  - {src: doc_a, src_group: category_rules, tgt: doc_b, type: INHERITS_GENERAL}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_and_validate_graph_config(bad_both_src)


def test_clause_graph_expands_without_cue_keywords():
    c_apparel = _chunk("jd_cat_07_fashion_apparel", "c1", "服饰规范", "吊牌规则")
    c_general = _chunk(
        "jd_rule_01_seven_days_general",
        "c2",
        "通用退货总则",
        "自签收次日零时起算",
        doc_group="general_rules",
    )
    cg = ClauseDependencyGraph([c_apparel, c_general])
    # Query intentionally avoids legacy keyword cues
    expanded = cg.expand_candidates("外衣吊牌拆了怎么办", [c_apparel])
    assert any(logical_doc_id_of(c) == "jd_rule_01_seven_days_general" for c in expanded)


@pytest.mark.asyncio
async def test_ce_threshold_gating_accepts_above_tau_and_rejects_below_tau():
    provider = _FixedScoreRerankProvider(
        {
            "直召1": 0.95,
            "直召2": 0.85,
            "直召3": 0.75,
            "直召4": 0.65,
            "直召5": 0.55,
            "图扩展低分": 0.02,
            "图扩展高分": 0.30,
        }
    )
    direct = [
        _chunk("doc_1", "d1", "直召1", "直召1"),
        _chunk("doc_2", "d2", "直召2", "直召2"),
        _chunk("doc_3", "d3", "直召3", "直召3"),
        _chunk("doc_4", "d4", "直召4", "直召4"),
        _chunk("doc_5", "d5", "直召5", "直召5"),
    ]

    # Below threshold (0.02 < 0.10) -> must NOT displace direct hits
    reranker_strict = DocumentReranker(provider, graph_ce_threshold=0.10)
    low_graph = _chunk("doc_g", "g_low", "图扩展低分", "图扩展低分", graph_expanded=True)
    res_low = await reranker_strict.rerank("q", [*direct, low_graph], top_n=5)
    assert [c.chunk_id for c in res_low] == ["d1", "d2", "d3", "d4", "d5"]

    # Above threshold (0.30 >= 0.10) -> injected into complementary slot 4 while Top-3 direct preserved
    high_graph = _chunk("doc_g", "g_high", "图扩展高分", "图扩展高分", graph_expanded=True)
    res_high = await reranker_strict.rerank("q", [*direct, high_graph], top_n=5)
    assert [c.chunk_id for c in res_high[:3]] == ["d1", "d2", "d3"]
    assert "g_high" in [c.chunk_id for c in res_high]


def test_suggest_edges_on_real_corpus():
    suggestions = extract_candidate_edges("knowledge_jd", config_path="config/policy_graph.yaml")
    assert isinstance(suggestions, list)
    assert any(s["tgt"] == "jd_ship_01_freight_standard" for s in suggestions)
