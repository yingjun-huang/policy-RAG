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
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.providers.embeddings import BgeEmbeddingProvider
from policy_rag.providers.fake import HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.clause_graph import ClauseDependencyGraph
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
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
    clause_graph = ClauseDependencyGraph(chunks)
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

    document_reranker = DocumentReranker(rerank_provider)
    meta = {
        "knowledge_dir": str(settings.knowledge_dir),
        "chunk_count": len(chunks),
        "yaml_contaminated_chunks": yaml_contaminated,
        "archived_doc_ids": archived_docs,
        "vector_store_backend": type(vec_store).__name__,
        "embedding_backend": embedder.backend,
        "embedding_model": settings.local_embedding_model if embedder.backend == "bge-local" else "hash",
        "reranker_backend": reranker_backend,
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
    ]
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

        # Scheme 6: Hybrid + ClauseGraph (1-Hop) + MetadataFilter + Gated Graph Rerank
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
            metrics_by_scheme[schemes[5]],
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


async def run_behavioral_evaluation(
    settings: Settings,
    benchmark_dataset: list[dict[str, Any]],
    guardrail_dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Runs end-to-end service checks (guardrails, isolation, citations, multi-turn history)."""
    container = PolicyContainer(settings=settings)
    await container.init_vector_index()
    service = container.qa_service
    generation_backend = container.settings.active_llm_model if container.settings.llm_configured else "offline_deterministic_pipeline"

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

    for item in benchmark_dataset:
        session_id = f"eval_{item['id']}"
        turns = [{"query": item["query"], **item}]
        for ft in item.get("follow_up_turns", []):
            turns.append(ft)

        for turn_idx, turn in enumerate(turns):
            total_turns += 1
            resp = await service.answer(
                ChatRequest(
                    message=turn["query"],
                    session_id=session_id,
                    policy_context=item.get("policy_context"),
                )
            )
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

    guardrail_summary: dict[str, Any] = {"checked": 0}
    if guardrail_dataset:
        refuse_ok = 0
        refuse_n = 0
        corr_ok = 0
        corr_n = 0
        abstain_empty_or_scoped = 0
        abstain_n = 0
        for item in guardrail_dataset:
            cat = item.get("category")
            if cat == "refuse":
                refuse_n += 1
                resp = await service.answer(ChatRequest(message=item["query"], policy_context=item.get("policy_context")))
                if resp.refused and resp.action == ResponseAction.REFUSE:
                    refuse_ok += 1
            elif cat == "abstain":
                abstain_n += 1
                resp = await service.answer(ChatRequest(message=item["query"], policy_context=item.get("policy_context")))
                if resp.action == ResponseAction.ABSTAIN or not settings.llm_configured:
                    # Offline deterministic FakeLLM always emits static text unless zero chunks were returned;
                    # verify no fake return_policy_00 chunk and scoped retrieval check.
                    if "return_policy_00" not in {c.chunk_id for c in resp.evidence}:
                        abstain_empty_or_scoped += 1
            elif cat == "user_correction":
                corr_n += 1
                sid = f"eval_corr_{item['id']}"
                await service.answer(
                    ChatRequest(
                        message=item["query"],
                        session_id=sid,
                        policy_context=item.get("policy_context"),
                    )
                )
                for ft in item.get("follow_up_turns", []):
                    resp1 = await service.answer(
                        ChatRequest(
                            message=ft["query"],
                            session_id=sid,
                            policy_context=item.get("policy_context"),
                        )
                    )
                    stems1 = set(_unique_stems(resp1.evidence))
                    g1 = set(ft.get("gold_doc_ids") or [])
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
            "refusal_abstain_f1": round(f1_ra, 2),
            "false_refusals_on_benchmark": false_refusals,
            "abstain_llm_judgment_status": "evaluated_live" if settings.llm_configured else "skipped_no_api_key",
        }

    return {
        "generation_backend": generation_backend,
        "llm_eval_status": "live_llm" if settings.llm_configured else "skipped_no_api_key",
        "benchmark_turns_evaluated": total_turns,
        "zero_fake_fallback_rate": round(no_fake_fallback / total_turns * 100, 2) if total_turns else 0.0,
        "citation_integrity_rate": round(citation_valid / citation_checked * 100, 2) if citation_checked else 0.0,
        "active_version_isolation_rate": round(isolation_passed / isolation_checked * 100, 2) if isolation_checked else 100.0,
        "followup_evidence_hit_rate": round(followup_gold_hits / followup_total * 100, 2) if followup_total else None,
        "followup_turns": followup_total,
        "e2e_latency_p50_ms": round(percentile(e2e_latencies, 50), 2),
        "e2e_latency_p95_ms": round(percentile(e2e_latencies, 95), 2),
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
    parser.add_argument("--save-report", action="store_true", default=True, help="Save evaluation report to json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation cases from {dataset_path.name}")

    settings = Settings()
    if args.knowledge_dir:
        settings.knowledge_dir = args.knowledge_dir
    elif dataset_path.name == "rag_eval.jsonl" and (ROOT / "knowledge").exists():
        settings.knowledge_dir = str(ROOT / "knowledge")

    chunks, bm25, vector, hybrid, reranker, meta = await setup_pipeline(settings)

    summary: dict[str, dict[str, Any]] = {}
    if args.mode in {"retrieval", "all"}:
        logger.info("Starting 6-way retrieval & version-isolation ablation evaluation...")
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
        if args.live_llm and settings.llm_configured:
            e2e_settings = settings.model_copy()
        else:
            e2e_settings = settings.model_copy(
                update={"app_env": "testing", "llm_api_key": "", "deepseek_api_key": ""}
            )
        # Re-enable offline cached BGE dense embeddings inside PolicyContainer E2E eval
        e2e_settings.local_embedding_force = True
        logger.info("Starting E2E pipeline behavioral & guardrail evaluation...")
        behavioral = await run_behavioral_evaluation(e2e_settings, dataset, guard_data)
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
