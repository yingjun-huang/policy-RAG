"""F4 Benchmark: Compare BGE embedding models (small vs base vs m3) on retrieval quality and CPU latency."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import complete_evidence_at_k, mrr, percentile, precision_at_k, recall_at_k
from eval.runner import _unique_stems, load_dataset
from policy_rag.config import Settings
from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.providers.embeddings import BgeEmbeddingProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.clause_graph import ClauseDependencyGraph
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import VectorRecord, VectorRetriever, build_vector_store

DEFAULT_MODELS = [
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-m3",
]


class _LocalOnnxBgeM3Provider:
    """Offline ONNX Runtime loader for locally cached BAAI/bge-m3 (onnx/model.onnx)."""

    def __init__(self, onnx_dir: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.session = ort.InferenceSession(
            str(onnx_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.dim = 1024
        self.backend = "bge-m3-onnx"
        self.available = True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        vecs: list[list[float]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            outputs = self.session.run(
                None,
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                },
            )
            emb = np.asarray(outputs[1], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            emb = emb / norms
            vecs.extend(emb.tolist())
        return vecs


def _find_cached_bge_m3_onnx() -> Path | None:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3" / "snapshots"
    if not hub_dir.exists():
        return None
    for snap in hub_dir.iterdir():
        onnx_dir = snap / "onnx"
        if (onnx_dir / "model.onnx").exists():
            return onnx_dir
    return None


async def benchmark_embedding_model(
    model_name: str,
    chunks: list[Any],
    bm25: BM25Retriever,
    clause_graph: ClauseDependencyGraph,
    reranker: DocumentReranker,
    dataset: list[dict[str, Any]],
    *,
    offline_only: bool = False,
) -> dict[str, Any]:
    if offline_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)

    settings = Settings(
        local_embedding_enabled=True,
        local_embedding_force=True,
        local_embedding_model=model_name,
        hf_endpoint="https://hf-mirror.com",
    )
    embedder: Any = BgeEmbeddingProvider(settings)
    if not embedder.available:
        if model_name == "BAAI/bge-m3":
            onnx_dir = _find_cached_bge_m3_onnx()
            if onnx_dir is not None:
                embedder = _LocalOnnxBgeM3Provider(onnx_dir)
            else:
                return {"model": model_name, "status": "unavailable", "error": embedder._load_error}
        else:
            return {"model": model_name, "status": "unavailable", "error": embedder._load_error}

    t_ingest = time.perf_counter()
    texts = [f"[{c.title} - {c.section or '条款'}] {c.content}" for c in chunks]
    embeddings = await embedder.embed(texts)
    ingest_ms = (time.perf_counter() - t_ingest) * 1000

    store = build_vector_store(
        backend="qdrant",
        collection=f"bench_{model_name.replace('/', '_')}",
        dim=int(embedder.dim),
    )
    store.upsert([VectorRecord(chunk=c, embedding=e) for c, e in zip(chunks, embeddings)])

    vec_ret = VectorRetriever(store, embedder)
    hyb_ret = HybridRetriever(bm25, vec_ret, rrf_k=settings.rrf_k, clause_graph=clause_graph)

    vec_p1, vec_r5, vec_comp5, vec_mrr, vec_lat = [], [], [], [], []
    full_p1, full_r5, full_comp5, full_ccomp5, full_mrr, full_lat = [], [], [], [], [], []

    for item in dataset:
        q = item["query"]
        gold = set(item.get("relevant_doc_ids") or item.get("gold_doc_ids") or [])
        if not gold:
            continue
        is_complex = item.get("category") in {"multi_hop", "table_lookup", "version_contrast"} or len(gold) >= 2
        flt = infer_retrieval_filter(
            q,
            default_as_of=settings.default_as_of,
            policy_context=item.get("policy_context"),
        )

        # 1. Filtered Vector-only search (isolates embedding model contribution)
        t0 = time.perf_counter()
        v_res = await vec_ret.retrieve(q, top_k=10, metadata_filter=flt)
        vec_lat.append((time.perf_counter() - t0) * 1000)
        v_stems = _unique_stems(v_res)
        vec_p1.append(precision_at_k(gold, v_stems, 1))
        vec_r5.append(recall_at_k(gold, v_stems, 5))
        vec_comp5.append(complete_evidence_at_k(gold, v_stems, 5))
        vec_mrr.append(mrr(gold, v_stems))

        # 2. Full deployed pipeline (Hybrid + Graph + Filter + Rerank)
        t0 = time.perf_counter()
        cands = await hyb_ret.retrieve(q, top_k=10, metadata_filter=flt, enable_graph_expansion=True)
        ranked = await reranker.rerank(q, cands, top_n=8)
        full_lat.append((time.perf_counter() - t0) * 1000)
        f_stems = _unique_stems(ranked)
        full_p1.append(precision_at_k(gold, f_stems, 1))
        full_r5.append(recall_at_k(gold, f_stems, 5))
        comp = complete_evidence_at_k(gold, f_stems, 5)
        full_comp5.append(comp)
        if is_complex:
            full_ccomp5.append(comp)
        full_mrr.append(mrr(gold, f_stems))

    n = len(vec_p1)
    cn = len(full_ccomp5)
    return {
        "model": model_name,
        "status": "ok",
        "backend": getattr(embedder, "backend", "bge-local"),
        "dim": int(embedder.dim),
        "index_embed_ms": round(ingest_ms, 2),
        "vector_only_filtered": {
            "precision@1": round(sum(vec_p1) / n * 100, 2),
            "recall@5": round(sum(vec_r5) / n * 100, 2),
            "complete@5": round(sum(vec_comp5) / n * 100, 2),
            "mrr": round(sum(vec_mrr) / n, 4),
            "latency_p50_ms": round(percentile(vec_lat, 50), 2),
            "latency_p95_ms": round(percentile(vec_lat, 95), 2),
        },
        "full_pipeline": {
            "precision@1": round(sum(full_p1) / n * 100, 2),
            "recall@5": round(sum(full_r5) / n * 100, 2),
            "complete@5": round(sum(full_comp5) / n * 100, 2),
            "complex_complete@5": round(sum(full_ccomp5) / cn * 100, 2) if cn else 0.0,
            "mrr": round(sum(full_mrr) / n, 4),
            "latency_p50_ms": round(percentile(full_lat, 50), 2),
            "latency_p95_ms": round(percentile(full_lat, 95), 2),
        },
    }


async def run_sweep(models: list[str], *, offline_only: bool = False) -> dict[str, Any]:
    if offline_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    settings = Settings()
    chunks = load_and_chunk_knowledge(settings.knowledge_dir)
    bm25 = BM25Retriever(chunks)
    clause_graph = ClauseDependencyGraph(chunks, config_path=settings.graph_config_path)
    reranker_provider = CrossEncoderRerankProvider(settings)
    if reranker_provider.backend == "lexical":
        raise RuntimeError("CrossEncoderRerankProvider fell back to lexical; offline ONNX/PyTorch reranker required.")
    reranker = DocumentReranker(reranker_provider, graph_ce_threshold=settings.graph_ce_threshold)
    dataset = load_dataset(ROOT / "eval" / "dataset" / "jd_eval_benchmark.jsonl")

    results = []
    for model_name in models:
        print(f"Evaluating embedding model: {model_name} ...")
        res = await benchmark_embedding_model(
            model_name,
            chunks,
            bm25,
            clause_graph,
            reranker,
            dataset,
            offline_only=offline_only,
        )
        results.append(res)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    return {
        "dataset_size": len(dataset),
        "reranker_backend": reranker_provider.backend,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BGE embedding models on policy-rag retrieval")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--offline-only", action="store_true", default=False)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "eval" / "reports" / "embedding_sweep.json"),
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    report = asyncio.run(run_sweep(args.models, offline_only=args.offline_only))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved embedding sweep report to {out_path}")


if __name__ == "__main__":
    main()
