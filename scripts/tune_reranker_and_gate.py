"""Offline T1 (CE threshold tau sweep) & T3 (Top-N & ONNX speedup sweep) benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.metrics import (
    complete_evidence_at_k,
    hit_at_k,
    mrr,
    percentile,
    precision_at_k,
    recall_at_k,
)
from eval.runner import _unique_stems, load_dataset, setup_pipeline
from policy_rag.config import Settings
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.reranker import DocumentReranker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "eval" / "dataset" / "jd_eval_benchmark.jsonl"
REPORTS_DIR = ROOT / "eval" / "reports"


def _summarize_metrics(
    records: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    n = len(records)
    complex_recs = [r for r in records if r["is_complex"]]
    return {
        "samples": n,
        "complex_samples": len(complex_recs),
        "precision@1": round(sum(r["p1"] for r in records) / n * 100, 2) if n else 0.0,
        "recall@3": round(sum(r["r3"] for r in records) / n * 100, 2) if n else 0.0,
        f"recall@{top_k}": round(sum(r["r5"] for r in records) / n * 100, 2) if n else 0.0,
        f"complete@{top_k}": round(sum(r["comp5"] for r in records) / n * 100, 2) if n else 0.0,
        f"complex_complete@{top_k}": (
            round(sum(r["comp5"] for r in complex_recs) / len(complex_recs) * 100, 2)
            if complex_recs
            else 0.0
        ),
        "hit@1": round(sum(r["hit1"] for r in records) / n * 100, 2) if n else 0.0,
        "hit@3": round(sum(r["hit3"] for r in records) / n * 100, 2) if n else 0.0,
        "mrr": round(sum(r["mrr"] for r in records) / n, 4) if n else 0.0,
        "latency_p50_ms": round(percentile([r["latency_ms"] for r in records], 50), 2),
        "latency_p95_ms": round(percentile([r["latency_ms"] for r in records], 95), 2),
    }


def _eval_stems(
    relevant_docs: set[str],
    stems: list[str],
    *,
    is_complex: bool,
    latency_ms: float,
    top_k: int = 5,
) -> dict[str, Any]:
    return {
        "is_complex": is_complex,
        "p1": precision_at_k(relevant_docs, stems, 1),
        "r3": recall_at_k(relevant_docs, stems, 3),
        "r5": recall_at_k(relevant_docs, stems, top_k),
        "comp5": complete_evidence_at_k(relevant_docs, stems, top_k),
        "hit1": hit_at_k(relevant_docs, stems, 1),
        "hit3": hit_at_k(relevant_docs, stems, 3),
        "mrr": mrr(relevant_docs, stems),
        "latency_ms": latency_ms,
    }


async def run_sweeps(limit: int | None = None) -> tuple[Path, Path]:
    settings = Settings()
    settings.local_embedding_force = True
    settings.cross_encoder_force = True

    _, _, _, hybrid, _, _ = await setup_pipeline(settings)
    dataset = load_dataset(DEFAULT_DATASET)
    if limit is not None:
        dataset = dataset[:limit]

    top_k = 5

    # Precompute candidates once so tau sweep is deterministic and fast
    prepared: list[dict[str, Any]] = []
    for item in dataset:
        q = item["query"]
        relevant_docs = set(item.get("relevant_doc_ids") or item.get("gold_doc_ids") or [])
        if not relevant_docs:
            continue
        is_complex = item.get("category") in {"multi_hop", "table_lookup", "version_contrast"} or len(relevant_docs) >= 2
        mf = infer_retrieval_filter(q, default_as_of=settings.default_as_of, policy_context=item.get("policy_context"))

        t0 = time.perf_counter()
        direct_10 = await hybrid.retrieve(q, top_k=10, metadata_filter=mf, enable_graph_expansion=False)
        filt_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        graph_cands = await hybrid.retrieve(q, top_k=10, metadata_filter=mf, enable_graph_expansion=True)
        graph_ret_ms = (time.perf_counter() - t1) * 1000

        prepared.append(
            {
                "query": q,
                "relevant_docs": relevant_docs,
                "is_complex": is_complex,
                "direct_10": direct_10,
                "filt_ms": filt_ms,
                "graph_cands": graph_cands,
                "graph_ret_ms": graph_ret_ms,
            }
        )

    # --- T1: Tau sweep on PyTorch Cross-Encoder ---
    pt_provider = CrossEncoderRerankProvider(settings, use_onnx=False)
    # Warmup
    await pt_provider.rerank("测试退货", ["七天无理由退货规则"], top_n=1)

    # Cache raw CE scores per query so we evaluate all thresholds without re-running transformer 15 times
    cached_scores: list[tuple[dict[str, Any], list[tuple[int, float]], float]] = []
    for row in prepared:
        cands = row["graph_cands"]
        texts = [f"[{c.title} - {c.section or '条款'}] {c.content}" for c in cands]
        t0 = time.perf_counter()
        ranked = await pt_provider.rerank(row["query"], texts, top_n=len(cands))
        ce_ms = (time.perf_counter() - t0) * 1000
        cached_scores.append((row, ranked, ce_ms))

    class _PrecomputedProvider:
        backend = "cross-encoder-precomputed"
        available = True

        def __init__(self, scores: list[tuple[int, float]]) -> None:
            self._scores = scores

        async def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
            return self._scores[:top_n]

    taus = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]
    tau_results: list[dict[str, Any]] = []
    for tau in taus:
        recs: list[dict[str, Any]] = []
        injected_counts: list[int] = []
        for row, ranked, ce_ms in cached_scores:
            t0 = time.perf_counter()
            reranker = DocumentReranker(_PrecomputedProvider(ranked), graph_ce_threshold=tau)
            selected = await reranker.rerank(row["query"], row["graph_cands"], top_k=top_k)
            gate_ms = (time.perf_counter() - t0) * 1000
            injected = sum(1 for c in selected if (c.metadata or {}).get("graph_expanded"))
            injected_counts.append(injected)
            recs.append(
                _eval_stems(
                    row["relevant_docs"],
                    _unique_stems(selected),
                    is_complex=row["is_complex"],
                    latency_ms=row["graph_ret_ms"] + ce_ms + gate_ms,
                    top_k=top_k,
                )
            )
        summary = _summarize_metrics(recs, top_k=top_k)
        summary["tau"] = tau
        summary["avg_graph_injected_in_top5"] = round(sum(injected_counts) / len(injected_counts), 3)
        tau_results.append(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gate_path = REPORTS_DIR / "rerank_gate_sweep.json"
    gate_payload = {
        "dataset": str(DEFAULT_DATASET),
        "samples": len(prepared),
        "reranker_backend": pt_provider.backend,
        "tau_sweep": tau_results,
    }
    gate_path.write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- T3: Top-N truncation & ONNX/int8 speedup sweep ---
    speed_variants: list[tuple[str, bool, bool, list[int]]] = [
        ("pytorch_fp32", False, False, [6, 8, 10]),
        ("onnx_fp32", True, False, [6, 8, 10]),
        ("onnx_int8", True, True, [6, 8, 10]),
    ]
    speed_results: list[dict[str, Any]] = []

    for label, use_onnx, onnx_int8, n_list in speed_variants:
        provider = CrossEncoderRerankProvider(settings, use_onnx=use_onnx, onnx_int8=onnx_int8)
        # Warmup
        await provider.rerank("测试退货", ["七天无理由退货规则", "偏远地区运费"], top_n=2)
        reranker = DocumentReranker(provider, graph_ce_threshold=0.02)

        for n_cand in n_list:
            recs = []
            rerank_only_latencies: list[float] = []
            for row in prepared:
                # Truncate direct RRF pool to N, then re-attach graph candidates?
                # Better: test pure RRF Top-N rerank (Scheme 5) AND graph-expanded with RRF Top-N (Scheme 6b)
                direct_n = row["direct_10"][:n_cand]
                t0 = time.perf_counter()
                reranked = await reranker.rerank(row["query"], direct_n, top_k=top_k)
                r_ms = (time.perf_counter() - t0) * 1000
                rerank_only_latencies.append(r_ms)
                recs.append(
                    _eval_stems(
                        row["relevant_docs"],
                        _unique_stems(reranked),
                        is_complex=row["is_complex"],
                        latency_ms=row["filt_ms"] + r_ms,
                        top_k=top_k,
                    )
                )
            summary = _summarize_metrics(recs, top_k=top_k)
            summary["runtime"] = label
            summary["backend"] = provider.backend
            summary["top_n_candidates"] = n_cand
            summary["rerank_only_p50_ms"] = round(percentile(rerank_only_latencies, 50), 2)
            summary["rerank_only_p95_ms"] = round(percentile(rerank_only_latencies, 95), 2)
            speed_results.append(summary)

    speed_path = REPORTS_DIR / "rerank_speed_sweep.json"
    speed_payload = {
        "dataset": str(DEFAULT_DATASET),
        "samples": len(prepared),
        "speed_sweep": speed_results,
    }
    speed_path.write_text(json.dumps(speed_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return gate_path, speed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    gate_path, speed_path = asyncio.run(run_sweeps(limit=args.limit))
    print(f"Saved: {gate_path}")
    print(f"Saved: {speed_path}")


if __name__ == "__main__":
    main()
