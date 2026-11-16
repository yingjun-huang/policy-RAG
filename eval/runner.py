"""Offline & E2E evaluation runner for retrieval ablation, version isolation, and pipeline invariants."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.judges import AnswerJudge
from eval.metrics import (
    complete_evidence_at_k,
    hit_at_k,
    mrr,
    no_archived_leak_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from policy_rag.config import Settings
from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.providers.base import LLMProvider
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.providers.embeddings import BgeEmbeddingProvider
from policy_rag.providers.fake import FakeLLMProvider, HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.providers.llm_cache import CachedLLMProvider, track_llm_usage
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.clause_graph import ClauseDependencyGraph
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.legacy import LegacyClauseDependencyGraph, LegacyDocumentReranker
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
    VectorRetriever,
    build_vector_store,
)
from policy_rag.schemas.chat import ChatRequest
from policy_rag.schemas.retrieval import ResponseAction, RetrievedChunk
from policy_rag.service import PolicyContainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.runner")

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "eval" / "dataset" / "jd_eval_benchmark.jsonl"
GUARDRAIL_DATASET_PATH = ROOT / "eval" / "dataset" / "jd_eval_guardrail_and_correction.jsonl"
REPORTS_DIR = ROOT / "eval" / "reports"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Loads JSONL evaluation queries and gold annotations."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _doc_stem(chunk: RetrievedChunk) -> str:
    """Extracts stable logical doc ID from chunk metadata or document_id."""
    logical_id = (chunk.metadata or {}).get("logical_doc_id")
    if logical_id:
        return str(logical_id)
    return chunk.document_id.rsplit("_", 1)[0]


def _unique_stems(chunks: list[RetrievedChunk]) -> list[str]:
    """Deduplicates chunk documents into unique logical document IDs preserving rank."""
    stems: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        stem = _doc_stem(chunk)
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


async def setup_pipeline(
    settings: Settings,
) -> tuple[
    list[RetrievedChunk],
    BM25Retriever,
    VectorRetriever,
    HybridRetriever,
    DocumentReranker,
    dict[str, Any],
]:
    logger.info(f"Loading and chunking knowledge from: {settings.knowledge_dir}")
    chunks = load_and_chunk_knowledge(settings.knowledge_dir)
    logger.info(f"Loaded {len(chunks)} knowledge chunks across policy documents")

    archived_docs = sorted(
        {
            _doc_stem(c)
            for c in chunks
            if str((c.metadata or {}).get("document_status", "current")).lower() == "archived"
        }
    )
    yaml_contaminated = sum(1 for c in chunks if c.content.lstrip().startswith("---"))

    bm25 = BM25Retriever(chunks)

    embedder = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        settings.local_embedding_force = True
        settings.cross_encoder_force = True
        embedder = BgeEmbeddingProvider(settings)
        if not embedder.available:
            raise RuntimeError("BGE model failed to load offline")
        logger.info(f"Initialized BGE Dense Embedder: {settings.local_embedding_model} (dim={embedder.dim})")
    except Exception as e:
        logger.warning(f"Could not load BGE model offline, falling back to HashEmbedding: {e}")
        embedder = HashEmbeddingProvider(dim=settings.embedding_dim)

    t0 = time.perf_counter()
    texts = [f"[{c.title} - {c.section or '条款'}] {c.content}" for c in chunks]
    embeddings = await embedder.embed(texts)
    vec_store = build_vector_store(
        backend=settings.vector_store_backend,
        collection=settings.qdrant_collection,
        dim=getattr(embedder, "dim", settings.embedding_dim),
        url=settings.qdrant_url,
        path=settings.qdrant_path,
    )
    records = [VectorRecord(chunk=c, embedding=emb) for c, emb in zip(chunks, embeddings)]
    vec_store.upsert(records)
    logger.info(
        f"Vector store ({type(vec_store).__name__}) populated with {len(vec_store)} chunks in {time.perf_counter() - t0:.2f}s"
    )

    vector_retriever = VectorRetriever(vec_store, embedder)
    clause_graph = ClauseDependencyGraph(chunks, config_path=settings.graph_config_path)
    hybrid_retriever = HybridRetriever(
        bm25,
        vector_retriever,
        rrf_k=settings.rrf_k,
        clause_graph=clause_graph,
    )

    try:
        rerank_provider = CrossEncoderRerankProvider(settings)
        reranker_backend = rerank_provider.backend
        logger.info(f"Initialized Cross-Encoder (backend={reranker_backend})")
    except Exception as e:
        logger.warning(f"Could not load Cross-Encoder, falling back to Lexical: {e}")
        rerank_provider = LexicalRerankProvider()
        reranker_backend = rerank_provider.backend

    document_reranker = DocumentReranker(
        rerank_provider,
        graph_ce_threshold=settings.graph_ce_threshold,
    )
    meta = {
        "knowledge_dir": str(settings.knowledge_dir),
        "chunk_count": len(chunks),
        "yaml_contaminated_chunks": yaml_contaminated,
        "archived_doc_ids": archived_docs,
        "vector_store_backend": type(vec_store).__name__,
        "embedding_backend": embedder.backend,
        "embedding_model": settings.local_embedding_model if embedder.backend == "bge-local" else "hash",
        "reranker_backend": reranker_backend,
        "cross_encoder_use_onnx": settings.cross_encoder_use_onnx,
        "faithfulness_risk_gate_enabled": settings.faithfulness_risk_gate_enabled,
        "clause_graph_nodes": clause_graph.node_count,
        "clause_graph_edges": clause_graph.edge_count,
    }
    return chunks, bm25, vector_retriever, hybrid_retriever, document_reranker, meta


def _record_scheme_metrics(
    bucket: dict[str, list[float]],
    *,
    relevant_docs: set[str],
    stems: list[str],
    latency_ms: float,
    top_k: int,
    archived_set: set[str],
    is_version_contrast: bool,
    is_complex: bool = False,
) -> None:
    bucket["latency_ms"].append(latency_ms)
    bucket["p1"].append(precision_at_k(relevant_docs, stems, 1))
    bucket["r3"].append(recall_at_k(relevant_docs, stems, 3))
    bucket["r5"].append(recall_at_k(relevant_docs, stems, top_k))
    comp = complete_evidence_at_k(relevant_docs, stems, top_k)
    bucket["complete5"].append(comp)
    if is_complex:
        bucket["complex_complete5"].append(comp)
    bucket["hit1"].append(hit_at_k(relevant_docs, stems, 1))
    bucket["hit3"].append(hit_at_k(relevant_docs, stems, 3))
    bucket["mrr"].append(mrr(relevant_docs, stems))
    if archived_set:
        if is_version_contrast:
            vc_ok = complete_evidence_at_k(relevant_docs, stems, top_k)
            bucket["vc_both"].append(vc_ok)
            bucket["version_match"].append(vc_ok)
        else:
            iso_ok = no_archived_leak_at_k(archived_set, stems, top_k)
            bucket["isolation"].append(iso_ok)
            bucket["version_match"].append(iso_ok)


async def run_ablation_evaluation(
    dataset: list[dict[str, Any]],
    bm25: BM25Retriever,
    vector: VectorRetriever,
    hybrid: HybridRetriever,
    reranker: DocumentReranker,
    *,
    archived_docs: list[str] | None = None,
    default_as_of: str = "2024-06-01T12:00:00+08:00",
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Runs systematic 6-way ablation across lexical, dense, hybrid, filtered, reranked, and ClauseGraph pipelines."""
    schemes = [
        "1. BM25 (无版本过滤)",
        "2. Vector (无版本过滤)",
        "3. Hybrid RRF (无版本过滤)",
        "4. Hybrid + MetadataFilter",
        "5. Hybrid + MetadataFilter + Rerank",
        "6. Hybrid + ClauseGraph (1-Hop) + MetadataFilter + Rerank",
        "6b. Hybrid + ClauseGraph (YAML, CE-Gated) + MetadataFilter + Rerank",
    ]
    legacy_graph = LegacyClauseDependencyGraph(bm25._chunks)
    legacy_hybrid = HybridRetriever(
        bm25,
        vector,
        rrf_k=hybrid.rrf_k,
        clause_graph=legacy_graph,  # type: ignore[arg-type]
    )
    legacy_reranker = LegacyDocumentReranker(reranker.provider)
    archived_set = set(archived_docs or [])

    metrics_by_scheme: dict[str, dict[str, list[float]]] = {
        s: {
            "p1": [],
            "r3": [],
            "r5": [],
            "complete5": [],
            "complex_complete5": [],
            "hit1": [],
            "hit3": [],
            "mrr": [],
            "isolation": [],
            "vc_both": [],
            "version_match": [],
            "latency_ms": [],
        }
        for s in schemes
    }

    for item in dataset:
        query = item["query"]
        relevant_docs = set(item.get("relevant_doc_ids") or item.get("gold_doc_ids") or [])
        if not relevant_docs:
            continue
        is_vc = item.get("category") == "version_contrast" or bool(relevant_docs & archived_set)
        is_complex = item.get("category") in {"multi_hop", "table_lookup", "version_contrast"} or len(relevant_docs) >= 2

        # Scheme 1: BM25 (unfiltered)
        t0 = time.perf_counter()
        bm25_res = await bm25.retrieve(query, top_k=top_k * 2)
        _record_scheme_metrics(
            metrics_by_scheme[schemes[0]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(bm25_res),
            latency_ms=(time.perf_counter() - t0) * 1000,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 2: Vector (unfiltered)
        t0 = time.perf_counter()
        vec_res = await vector.retrieve(query, top_k=top_k * 2)
        _record_scheme_metrics(
            metrics_by_scheme[schemes[1]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(vec_res),
            latency_ms=(time.perf_counter() - t0) * 1000,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 3: Hybrid (unfiltered, no graph expansion)
        t0 = time.perf_counter()
        hyb_res = await hybrid.retrieve(query, top_k=top_k * 2, enable_graph_expansion=False)
        _record_scheme_metrics(
            metrics_by_scheme[schemes[2]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(hyb_res),
            latency_ms=(time.perf_counter() - t0) * 1000,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 4: Hybrid + MetadataFilter (no graph expansion)
        metadata_filter = infer_retrieval_filter(
            query,
            default_as_of=default_as_of,
            policy_context=item.get("policy_context"),
        )
        t0 = time.perf_counter()
        filt_res = await hybrid.retrieve(
            query,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
            enable_graph_expansion=False,
        )
        filt_ms = (time.perf_counter() - t0) * 1000
        _record_scheme_metrics(
            metrics_by_scheme[schemes[3]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(filt_res),
            latency_ms=filt_ms,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 5: Hybrid + MetadataFilter + Rerank (no graph expansion)
        t0 = time.perf_counter()
        reranked = await reranker.rerank(query, filt_res, top_n=max(top_k, 8))
        rerank_ms = filt_ms + (time.perf_counter() - t0) * 1000
        _record_scheme_metrics(
            metrics_by_scheme[schemes[4]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(reranked),
            latency_ms=rerank_ms,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 6: Legacy Hybrid + ClauseGraph (1-Hop, cue-triggered) + MetadataFilter + Legacy Rerank
        t0 = time.perf_counter()
        legacy_cand = await legacy_hybrid.retrieve(
            query,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
            enable_graph_expansion=True,
        )
        legacy_reranked = await legacy_reranker.rerank(query, legacy_cand, top_n=max(top_k, 8))
        legacy_ms = (time.perf_counter() - t0) * 1000
        _record_scheme_metrics(
            metrics_by_scheme[schemes[5]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(legacy_reranked),
            latency_ms=legacy_ms,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

        # Scheme 6b: Hybrid + ClauseGraph (YAML) + MetadataFilter + CE-Gated Rerank
        t0 = time.perf_counter()
        graph_cand = await hybrid.retrieve(
            query,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
            enable_graph_expansion=True,
        )
        graph_reranked = await reranker.rerank(query, graph_cand, top_n=max(top_k, 8))
        graph_ms = (time.perf_counter() - t0) * 1000
        _record_scheme_metrics(
            metrics_by_scheme[schemes[6]],
            relevant_docs=relevant_docs,
            stems=_unique_stems(graph_reranked),
            latency_ms=graph_ms,
            top_k=top_k,
            archived_set=archived_set,
            is_version_contrast=is_vc,
            is_complex=is_complex,
        )

    summary: dict[str, dict[str, Any]] = {}
    for s in schemes:
        m = metrics_by_scheme[s]
        n = len(m["p1"])
        comp_n = len(m["complex_complete5"])
        iso_n = len(m["isolation"])
        vc_n = len(m["vc_both"])
        vm_n = len(m["version_match"])
        summary[s] = {
            "samples": n,
            "complex_samples": comp_n,
            "precision@1": round(sum(m["p1"]) / n * 100, 2) if n else 0.0,
            "recall@3": round(sum(m["r3"]) / n * 100, 2) if n else 0.0,
            f"recall@{top_k}": round(sum(m["r5"]) / n * 100, 2) if n else 0.0,
            f"complete@{top_k}": round(sum(m["complete5"]) / n * 100, 2) if n else 0.0,
            f"complex_complete@{top_k}": round(sum(m["complex_complete5"]) / comp_n * 100, 2) if comp_n else 0.0,
            "hit@1": round(sum(m["hit1"]) / n * 100, 2) if n else 0.0,
            "hit@3": round(sum(m["hit3"]) / n * 100, 2) if n else 0.0,
            "mrr": round(sum(m["mrr"]) / n, 4) if n else 0.0,
            "version_isolation_rate": round(sum(m["isolation"]) / iso_n * 100, 2) if iso_n else None,
            "version_contrast_both_rate": round(sum(m["vc_both"]) / vc_n * 100, 2) if vc_n else None,
            "version_match_accuracy": round(sum(m["version_match"]) / vm_n * 100, 2) if vm_n else None,
            "latency_p50_ms": round(percentile(m["latency_ms"], 50), 2),
            "latency_p95_ms": round(percentile(m["latency_ms"], 95), 2),
        }

    return summary


def _usage_snap(llm: Any) -> dict[str, int]:
    if hasattr(llm, "usage_snapshot"):
        snap = llm.usage_snapshot()
        return {
            "calls": int(snap.get("calls", 0)),
            "cache_hits": int(snap.get("cache_hits", 0)),
            "prompt_tokens": int(snap.get("prompt_tokens", 0)),
            "completion_tokens": int(snap.get("completion_tokens", 0)),
            "total_tokens": int(snap.get("total_tokens", 0)),
        }
    return {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _expected_route_for_case(category: str | None) -> str | None:
    """Maps unambiguous dataset categories to gold Adaptive-RAG retrieval routes."""
    if category in {"single_hop", "table_lookup"}:
        return "single"
    if category in {"multi_hop", "version_contrast"}:
        return "multi"
    if category == "refuse":
        return "refuse"
    return None


async def run_behavioral_evaluation(
    settings: Settings,
    benchmark_dataset: list[dict[str, Any]],
    guardrail_dataset: list[dict[str, Any]] | None = None,
    *,
    live_llm: bool = False,
    judge_llm: LLMProvider | None = None,
    concurrency: int = 6,
    predictions_out: Path | None = None,
) -> dict[str, Any]:
    """Runs end-to-end service checks (guardrails, judge, faithfulness, per-route latency & token usage)."""
    container = PolicyContainer(settings=settings)
    await container.init_vector_index()
    service = container.qa_service
    is_live = bool(live_llm and container.settings.llm_configured)
    generation_backend = container.settings.active_llm_model if is_live else "offline_deterministic_pipeline"
    effective_concurrency = max(1, int(concurrency))

    if judge_llm is not None:
        active_judge_llm: LLMProvider = judge_llm
        judge_backend = str(getattr(judge_llm, "model", "custom-judge"))
    elif is_live:
        raw_judge = OpenAICompatibleLLMProvider(
            api_key=settings.active_judge_api_key,
            base_url=settings.active_judge_base_url,
            model=settings.active_judge_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        active_judge_llm = CachedLLMProvider(
            raw_judge,
            db_path=settings.llm_cache_db_path,
            model_name=settings.active_judge_model,
            budget_tokens=settings.llm_token_budget,
            enabled=settings.llm_cache_enabled,
        )
        judge_backend = settings.active_judge_model
    else:
        active_judge_llm = CachedLLMProvider(
            FakeLLMProvider(default_response='{"verdict": "partial", "rationale": "offline fake judge"}'),
            model_name="fake-judge",
            enabled=False,
        )
        judge_backend = "fake-judge"

    judge = AnswerJudge(active_judge_llm)

    archived_set = {
        _doc_stem(c)
        for c in container.chunks
        if str((c.metadata or {}).get("document_status", "current")).lower() == "archived"
    }

    total_turns = 0
    citation_valid = 0
    citation_checked = 0
    no_fake_fallback = 0
    isolation_passed = 0
    isolation_checked = 0
    followup_gold_hits = 0
    followup_total = 0
    false_refusals = 0
    e2e_latencies: list[float] = []

    route_stats: dict[str, dict[str, Any]] = {}

    def _record_route(route_name: str, resp: Any, delta: dict[str, int]) -> None:
        bucket = route_stats.setdefault(
            route_name,
            {
                "turns": 0,
                "latencies_ms": [],
                "retrieval_ms": [],
                "rerank_ms": [],
                "llm_ms": [],
                "faithfulness_ms": [],
                "llm_calls": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        bucket["turns"] += 1
        bucket["latencies_ms"].append(resp.latency.total_ms)
        bucket["retrieval_ms"].append(resp.latency.retrieval_ms)
        bucket["rerank_ms"].append(resp.latency.rerank_ms)
        bucket["llm_ms"].append(resp.latency.llm_ms)
        bucket["faithfulness_ms"].append(resp.latency.faithfulness_ms)
        bucket["llm_calls"] += delta["calls"]
        bucket["cache_hits"] += delta["cache_hits"]
        bucket["prompt_tokens"] += delta["prompt_tokens"]
        bucket["completion_tokens"] += delta["completion_tokens"]
        bucket["total_tokens"] += delta["total_tokens"]

    judge_verdicts = {"correct": 0, "partial": 0, "incorrect": 0}
    judge_scores: list[float] = []

    clarify_expected = 0
    clarify_predicted = 0
    clarify_tp = 0

    route_eval_stats: dict[str, dict[str, int]] = {}
    prediction_rows: list[dict[str, Any]] = []

    faith_eligible = 0
    faith_checked = 0
    faith_triggered = 0
    faith_pre_scores: list[float] = []
    faith_post_scores: list[float] = []
    faith_miscorrections = 0
    faith_gate_reasons: dict[str, int] = {}
    faith_checked_ms: list[float] = []
    faith_eligible_ms: list[float] = []

    sem = asyncio.Semaphore(effective_concurrency)
    total_items = len(benchmark_dataset)
    completed_items = 0

    async def _run_benchmark_item(item: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal completed_items
        session_id = f"eval_{item['id']}"
        turns = [{"query": item["query"], **item}]
        for ft in item.get("follow_up_turns", []):
            turns.append(ft)

        item_records: list[dict[str, Any]] = []
        async with sem:
            for turn_idx, turn in enumerate(turns):
                with track_llm_usage() as turn_usage:
                    resp = await service.answer(
                        ChatRequest(
                            message=turn["query"],
                            session_id=session_id,
                            policy_context=item.get("policy_context"),
                        )
                    )

                route_name = resp.strategy.value if hasattr(resp.strategy, "value") else str(resp.strategy)
                expected_action = str(turn.get("expected_action") or "answer").lower()
                pred_action = resp.action.value if hasattr(resp.action, "value") else str(resp.action)

                post_j = None
                pre_j = None
                gold_answer = str(turn.get("gold_answer") or "")
                if gold_answer.strip():
                    post_j = await judge.evaluate(
                        query=turn["query"],
                        gold_answer=gold_answer,
                        predicted_answer=resp.answer,
                        expected_action=expected_action,
                        predicted_action=pred_action,
                    )
                    if resp.faithfulness_triggered and resp.pre_correction_answer:
                        pre_j = await judge.evaluate(
                            query=turn["query"],
                            gold_answer=gold_answer,
                            predicted_answer=resp.pre_correction_answer,
                            expected_action=expected_action,
                            predicted_action=pred_action,
                        )

                item_records.append(
                    {
                        "item": item,
                        "turn": turn,
                        "turn_idx": turn_idx,
                        "resp": resp,
                        "usage": turn_usage,
                        "route_name": route_name,
                        "expected_action": expected_action,
                        "pred_action": pred_action,
                        "post_j": post_j,
                        "pre_j": pre_j,
                    }
                )

        completed_items += 1
        if completed_items % 20 == 0 or completed_items == total_items:
            logger.info(
                "Behavioral benchmark progress: %d/%d items (concurrency=%d)",
                completed_items,
                total_items,
                effective_concurrency,
            )
        return item_records

    all_item_records = await asyncio.gather(*[_run_benchmark_item(item) for item in benchmark_dataset])

    for item_records in all_item_records:
        for rec in item_records:
            total_turns += 1
            item = rec["item"]
            turn = rec["turn"]
            turn_idx = rec["turn_idx"]
            resp = rec["resp"]
            delta = rec["usage"]
            route_name = rec["route_name"]
            expected_action = rec["expected_action"]
            pred_action = rec["pred_action"]
            post_j = rec["post_j"]
            pre_j = rec["pre_j"]

            _record_route(route_name, resp, delta)

            e2e_latencies.append(resp.latency.total_ms)
            if resp.refused:
                false_refusals += 1
            ev_ids = {c.chunk_id for c in resp.evidence}
            if "return_policy_00" not in ev_ids:
                no_fake_fallback += 1

            if resp.evidence:
                citation_checked += 1
                if resp.citations and all(c.chunk_id in ev_ids for c in resp.citations):
                    citation_valid += 1

            stems = _unique_stems(resp.evidence)
            gdocs = set(turn.get("gold_doc_ids") or item.get("gold_doc_ids") or [])
            if item.get("category") != "version_contrast" and not (gdocs & archived_set):
                isolation_checked += 1
                if not any(s in archived_set for s in stems):
                    isolation_passed += 1

            if turn_idx > 0 and gdocs:
                followup_total += 1
                if gdocs.issubset(set(stems)) or any(d in stems for d in gdocs):
                    followup_gold_hits += 1

            if expected_action == "clarify":
                clarify_expected += 1
            if pred_action == "clarify":
                clarify_predicted += 1
            if expected_action == "clarify" and pred_action == "clarify":
                clarify_tp += 1

            if resp.evidence and not resp.refused and route_name != "direct":
                faith_eligible += 1
                faith_eligible_ms.append(float(resp.latency.faithfulness_ms))
                gate_reason = resp.faithfulness_gate_reason or (
                    "checked" if resp.faithfulness_checked else "skipped"
                )
                faith_gate_reasons[gate_reason] = faith_gate_reasons.get(gate_reason, 0) + 1
                if resp.faithfulness_checked:
                    faith_checked += 1
                    faith_checked_ms.append(float(resp.latency.faithfulness_ms))
                if resp.faithfulness_triggered:
                    faith_triggered += 1

            if turn_idx == 0:
                exp_route = _expected_route_for_case(item.get("category"))
                if exp_route is not None:
                    rb = route_eval_stats.setdefault(exp_route, {"total": 0, "matched": 0})
                    rb["total"] += 1
                    if route_name == exp_route:
                        rb["matched"] += 1

            if post_j is not None:
                judge_verdicts[post_j.verdict] += 1
                judge_scores.append(post_j.score)
                prediction_rows.append(
                    {
                        "id": item.get("id"),
                        "category": item.get("category"),
                        "turn_idx": turn_idx,
                        "query": turn["query"],
                        "gold_answer": str(turn.get("gold_answer") or ""),
                        "expected_action": expected_action,
                        "predicted_action": pred_action,
                        "predicted_answer": resp.answer,
                        "route": route_name,
                        "faithfulness_checked": resp.faithfulness_checked,
                        "faithfulness_triggered": resp.faithfulness_triggered,
                        "primary_judge_verdict": post_j.verdict,
                        "primary_judge_score": post_j.score,
                    }
                )
                if pre_j is not None:
                    faith_pre_scores.append(pre_j.score)
                    faith_post_scores.append(post_j.score)
                    if pre_j.score > post_j.score:
                        faith_miscorrections += 1

    guardrail_summary: dict[str, Any] = {"checked": 0}
    if guardrail_dataset:
        async def _run_guardrail_item(item: dict[str, Any]) -> dict[str, Any]:
            cat = item.get("category")
            async with sem:
                if cat == "refuse":
                    with track_llm_usage() as turn_usage:
                        resp = await service.answer(
                            ChatRequest(message=item["query"], policy_context=item.get("policy_context"))
                        )
                    return {"cat": cat, "turns": [(resp, turn_usage)], "item": item}
                if cat == "abstain":
                    with track_llm_usage() as turn_usage:
                        resp = await service.answer(
                            ChatRequest(message=item["query"], policy_context=item.get("policy_context"))
                        )
                    return {"cat": cat, "turns": [(resp, turn_usage)], "item": item}
                if cat == "user_correction":
                    sid = f"eval_corr_{item['id']}"
                    recorded_turns: list[tuple[Any, dict[str, int], dict[str, Any] | None]] = []
                    with track_llm_usage() as turn_usage0:
                        resp0 = await service.answer(
                            ChatRequest(
                                message=item["query"],
                                session_id=sid,
                                policy_context=item.get("policy_context"),
                            )
                        )
                    recorded_turns.append((resp0, turn_usage0, None))
                    for ft in item.get("follow_up_turns", []):
                        with track_llm_usage() as turn_usage_ft:
                            resp1 = await service.answer(
                                ChatRequest(
                                    message=ft["query"],
                                    session_id=sid,
                                    policy_context=item.get("policy_context"),
                                )
                            )
                        recorded_turns.append((resp1, turn_usage_ft, ft))
                    return {"cat": cat, "turns": recorded_turns, "item": item}
                return {"cat": cat, "turns": [], "item": item}

        guard_records = await asyncio.gather(*[_run_guardrail_item(item) for item in guardrail_dataset])

        refuse_ok = 0
        refuse_n = 0
        corr_ok = 0
        corr_n = 0
        abstain_empty_or_scoped = 0
        abstain_action_ok = 0
        abstain_n = 0
        for rec in guard_records:
            cat = rec["cat"]
            if cat == "refuse":
                refuse_n += 1
                resp, turn_usage = rec["turns"][0]
                r_strat = resp.strategy.value if hasattr(resp.strategy, "value") else str(resp.strategy)
                _record_route(r_strat, resp, turn_usage)
                rb = route_eval_stats.setdefault("refuse", {"total": 0, "matched": 0})
                rb["total"] += 1
                if r_strat == "refuse":
                    rb["matched"] += 1
                if resp.refused and resp.action == ResponseAction.REFUSE:
                    refuse_ok += 1
            elif cat == "abstain":
                abstain_n += 1
                resp, turn_usage = rec["turns"][0]
                _record_route(
                    resp.strategy.value if hasattr(resp.strategy, "value") else str(resp.strategy),
                    resp,
                    turn_usage,
                )
                if resp.action == ResponseAction.ABSTAIN:
                    abstain_action_ok += 1
                if resp.action == ResponseAction.ABSTAIN or not is_live:
                    if "return_policy_00" not in {c.chunk_id for c in resp.evidence}:
                        abstain_empty_or_scoped += 1
            elif cat == "user_correction":
                corr_n += 1
                resp0, turn_usage0, _ = rec["turns"][0]
                _record_route(
                    resp0.strategy.value if hasattr(resp0.strategy, "value") else str(resp0.strategy),
                    resp0,
                    turn_usage0,
                )
                for resp1, turn_usage_ft, ft in rec["turns"][1:]:
                    _record_route(
                        resp1.strategy.value if hasattr(resp1.strategy, "value") else str(resp1.strategy),
                        resp1,
                        turn_usage_ft,
                    )
                    stems1 = set(_unique_stems(resp1.evidence))
                    g1 = set((ft or {}).get("gold_doc_ids") or [])
                    if g1.issubset(stems1):
                        corr_ok += 1

        tp = refuse_ok + abstain_empty_or_scoped
        fp = false_refusals
        fn = (refuse_n - refuse_ok) + (abstain_n - abstain_empty_or_scoped)
        precision_ra = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_ra = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ra = (2 * precision_ra * recall_ra / (precision_ra + recall_ra) * 100) if (precision_ra + recall_ra) > 0 else 0.0

        guardrail_summary = {
            "checked": len(guardrail_dataset),
            "refuse_accuracy": round(refuse_ok / refuse_n * 100, 2) if refuse_n else None,
            "refuse_passed": refuse_ok,
            "refuse_total": refuse_n,
            "user_correction_retrieval_accuracy": round(corr_ok / corr_n * 100, 2) if corr_n else None,
            "user_correction_passed": corr_ok,
            "user_correction_total": corr_n,
            "abstain_no_fake_fallback_rate": round(abstain_empty_or_scoped / abstain_n * 100, 2) if abstain_n else None,
            "abstain_action_accuracy": round(abstain_action_ok / abstain_n * 100, 2) if abstain_n else None,
            "abstain_action_passed": abstain_action_ok,
            "abstain_total": abstain_n,
            "refusal_abstain_f1": round(f1_ra, 2),
            "false_refusals_on_benchmark": false_refusals,
            "abstain_llm_judgment_status": "evaluated_live" if is_live else "offline_fake_llm",
        }

    by_route_summary: dict[str, Any] = {}
    for r_name, b in sorted(route_stats.items()):
        t_n = b["turns"]
        by_route_summary[r_name] = {
            "turns": t_n,
            "latency_p50_ms": round(percentile(b["latencies_ms"], 50), 2),
            "latency_p95_ms": round(percentile(b["latencies_ms"], 95), 2),
            "avg_retrieval_ms": round(sum(b["retrieval_ms"]) / t_n, 2) if t_n else 0.0,
            "avg_rerank_ms": round(sum(b["rerank_ms"]) / t_n, 2) if t_n else 0.0,
            "avg_llm_ms": round(sum(b["llm_ms"]) / t_n, 2) if t_n else 0.0,
            "avg_faithfulness_ms": round(sum(b["faithfulness_ms"]) / t_n, 2) if t_n else 0.0,
            "llm_calls": b["llm_calls"],
            "avg_llm_calls_per_turn": round(b["llm_calls"] / t_n, 2) if t_n else 0.0,
            "cache_hits": b["cache_hits"],
            "prompt_tokens": b["prompt_tokens"],
            "completion_tokens": b["completion_tokens"],
            "total_tokens": b["total_tokens"],
        }

    judge_n = len(judge_scores)
    clarify_precision = (clarify_tp / clarify_predicted * 100) if clarify_predicted else None
    clarify_recall = (clarify_tp / clarify_expected * 100) if clarify_expected else None

    if is_live:
        if hasattr(container.llm, "flush_ledger"):
            container.llm.flush_ledger("eval.runner.pipeline")
        if hasattr(active_judge_llm, "flush_ledger"):
            active_judge_llm.flush_ledger("eval.runner.judge")

    if predictions_out is not None and prediction_rows:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        predictions_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in prediction_rows) + "\n",
            encoding="utf-8",
        )

    route_eval_total = sum(v["total"] for v in route_eval_stats.values())
    route_eval_matched = sum(v["matched"] for v in route_eval_stats.values())
    all_route_turns = sum(b["turns"] for b in by_route_summary.values())
    predicted_dist_pct = {
        r: round(b["turns"] / all_route_turns * 100, 2)
        for r, b in by_route_summary.items()
    } if all_route_turns else {}

    return {
        "generation_backend": generation_backend,
        "judge_backend": judge_backend,
        "reranker_backend": getattr(container.reranker, "backend", type(container.reranker).__name__),
        "llm_eval_status": "live_llm" if is_live else "offline_fake_llm",
        "concurrency": effective_concurrency,
        "benchmark_turns_evaluated": total_turns,
        "zero_fake_fallback_rate": round(no_fake_fallback / total_turns * 100, 2) if total_turns else 0.0,
        "citation_integrity_rate": round(citation_valid / citation_checked * 100, 2) if citation_checked else 0.0,
        "active_version_isolation_rate": round(isolation_passed / isolation_checked * 100, 2) if isolation_checked else 100.0,
        "followup_evidence_hit_rate": round(followup_gold_hits / followup_total * 100, 2) if followup_total else None,
        "followup_turns": followup_total,
        "e2e_latency_p50_ms": round(percentile(e2e_latencies, 50), 2),
        "e2e_latency_p95_ms": round(percentile(e2e_latencies, 95), 2),
        "routing_metrics": {
            "evaluated": route_eval_total,
            "matched": route_eval_matched,
            "accuracy": round(route_eval_matched / route_eval_total * 100, 2) if route_eval_total else None,
            "by_expected_route": {
                k: {
                    **v,
                    "accuracy": round(v["matched"] / v["total"] * 100, 2) if v["total"] else None,
                }
                for k, v in sorted(route_eval_stats.items())
            },
            "predicted_distribution_pct": predicted_dist_pct,
        },
        "answer_correctness": {
            "evaluated": judge_n,
            "correct": judge_verdicts["correct"],
            "partial": judge_verdicts["partial"],
            "incorrect": judge_verdicts["incorrect"],
            "strict_accuracy": round(judge_verdicts["correct"] / judge_n * 100, 2) if judge_n else None,
            "weighted_accuracy": round(sum(judge_scores) / judge_n * 100, 2) if judge_n else None,
        },
        "clarification_metrics": {
            "expected_total": clarify_expected,
            "predicted_total": clarify_predicted,
            "matched_tp": clarify_tp,
            "precision": round(clarify_precision, 2) if clarify_precision is not None else None,
            "recall": round(clarify_recall, 2) if clarify_recall is not None else None,
        },
        "faithfulness_metrics": {
            "eligible_turns": faith_eligible,
            "checked_turns": faith_checked,
            "skipped_turns": faith_eligible - faith_checked,
            "audit_rate": round(faith_checked / faith_eligible * 100, 2) if faith_eligible else 0.0,
            "gate_reasons": faith_gate_reasons,
            "avg_eligible_faithfulness_ms": (
                round(sum(faith_eligible_ms) / len(faith_eligible_ms), 2) if faith_eligible_ms else 0.0
            ),
            "avg_checked_faithfulness_ms": (
                round(sum(faith_checked_ms) / len(faith_checked_ms), 2) if faith_checked_ms else 0.0
            ),
            "triggered_count": faith_triggered,
            "trigger_rate": round(faith_triggered / faith_checked * 100, 2) if faith_checked else 0.0,
            "pre_correction_weighted_accuracy": (
                round(sum(faith_pre_scores) / len(faith_pre_scores) * 100, 2) if faith_pre_scores else None
            ),
            "post_correction_weighted_accuracy": (
                round(sum(faith_post_scores) / len(faith_post_scores) * 100, 2) if faith_post_scores else None
            ),
            "miscorrection_count": faith_miscorrections,
            "miscorrection_rate": (
                round(faith_miscorrections / len(faith_pre_scores) * 100, 2) if faith_pre_scores else None
            ),
        },
        "by_route": by_route_summary,
        "llm_usage": {
            "pipeline": _usage_snap(container.llm),
            "judge": _usage_snap(active_judge_llm),
        },
        "guardrail_and_correction": guardrail_summary,
    }


def print_report_table(summary: dict[str, dict[str, Any]], dataset_size: int) -> str:
    lines = [
        "",
        f"### 离线检索与版本隔离消融实测报告（样本数 N = {dataset_size}，逻辑文档级去重口径）",
        "",
        "| 方案 (Scheme) | P@1 | R@3 | R@5 | Complete@5 | 复杂问题Complete@5 | MRR | 版本隔离率 | 跨版本双全率 | 版本总准确率 | P50 | P95 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s, m in summary.items():
        iso = f"{m['version_isolation_rate']}%" if m["version_isolation_rate"] is not None else "N/A"
        vc = f"{m['version_contrast_both_rate']}%" if m["version_contrast_both_rate"] is not None else "N/A"
        vm = f"{m.get('version_match_accuracy')}%" if m.get("version_match_accuracy") is not None else "N/A"
        lines.append(
            f"| **{s}** | {m['precision@1']}% | {m['recall@3']}% | {m.get('recall@5', 0.0)}% | "
            f"{m.get('complete@5', 0.0)}% | {m.get('complex_complete@5', 0.0)}% | {m['mrr']} | {iso} | {vc} | {vm} | "
            f"{m['latency_p50_ms']}ms | {m['latency_p95_ms']}ms |"
        )
    lines.append("")
    report_text = "\n".join(lines)
    print(report_text)
    return report_text


async def main() -> None:
    parser = argparse.ArgumentParser(description="Policy RAG Offline & E2E Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Path to jsonl evaluation dataset")
    parser.add_argument(
        "--guardrail-dataset",
        type=str,
        default=str(GUARDRAIL_DATASET_PATH),
        help="Path to guardrail/abstain/correction evaluation jsonl",
    )
    parser.add_argument("--knowledge-dir", type=str, default=None, help="Override knowledge directory path")
    parser.add_argument("--mode", choices=["retrieval", "e2e", "all"], default="all", help="Evaluation scope")
    parser.add_argument("--live-llm", action="store_true", default=False, help="Use live LLM API for E2E evaluation")
    parser.add_argument("--concurrency", type=int, default=6, help="Max concurrent evaluation items during E2E run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation items per dataset")
    parser.add_argument("--save-report", action="store_true", default=True, help="Save evaluation report to json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    logger.info(f"Loaded {len(dataset)} evaluation cases from {dataset_path.name}")

    settings = Settings()
    if args.knowledge_dir:
        settings.knowledge_dir = args.knowledge_dir
    elif dataset_path.name == "rag_eval.jsonl" and (ROOT / "knowledge").exists():
        settings.knowledge_dir = str(ROOT / "knowledge")

    chunks, bm25, vector, hybrid, reranker, meta = await setup_pipeline(settings)

    summary: dict[str, dict[str, Any]] = {}
    if args.mode in {"retrieval", "all"}:
        logger.info("Starting 7-way retrieval & version-isolation ablation evaluation...")
        summary = await run_ablation_evaluation(
            dataset,
            bm25,
            vector,
            hybrid,
            reranker,
            archived_docs=meta["archived_doc_ids"],
            default_as_of=settings.default_as_of,
        )
        print_report_table(summary, len(dataset))

    behavioral: dict[str, Any] | None = None
    if args.mode in {"e2e", "all"}:
        guard_path = Path(args.guardrail_dataset)
        guard_data = load_dataset(guard_path) if guard_path.exists() and dataset_path.name != "rag_eval.jsonl" else None
        if guard_data is not None and args.limit is not None:
            guard_data = guard_data[: args.limit]
        is_live = bool(args.live_llm and settings.llm_configured)
        if is_live:
            e2e_settings = settings.model_copy()
        else:
            e2e_settings = settings.model_copy(
                update={"app_env": "testing", "llm_api_key": "", "deepseek_api_key": ""}
            )
        # Re-enable offline cached BGE dense embeddings inside PolicyContainer E2E eval
        e2e_settings.local_embedding_force = True
        logger.info(
            "Starting E2E pipeline behavioral & guardrail evaluation (concurrency=%d)...",
            args.concurrency,
        )
        pred_path = REPORTS_DIR / "predictions_latest.jsonl" if (is_live and args.save_report and args.limit is None) else None
        behavioral = await run_behavioral_evaluation(
            e2e_settings,
            dataset,
            guard_data,
            live_llm=is_live,
            concurrency=args.concurrency,
            predictions_out=pred_path,
        )
        print("E2E Behavioral Summary:", json.dumps(behavioral, ensure_ascii=False, indent=2))

    if args.save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"report_{timestamp}.json"
        report_data = {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "dataset_size": len(dataset),
            "pipeline_metadata": meta,
            "summary": summary,
            "behavioral": behavioral,
        }
        report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved evaluation report to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
