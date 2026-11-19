"""Replay saved answers with actual histories; optionally rejudge both models.

No answer generation, retrieval, dataset edits, local cache writes or ledger writes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judges import AnswerJudge
from eval.judge_context import JUDGE_PROTOCOL_VERSION, hydrate_prediction_context, load_reference_corpus
from policy_rag.config import Settings
from policy_rag.providers.llm_cache import CachedLLMProvider
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider


async def cross_judge_predictions(
    predictions: list[dict[str, Any]], *, api_key: str, base_url: str,
    model: str, concurrency: int = 6, primary_model: str | None = None,
) -> dict[str, Any]:
    def provider(model_name: str) -> CachedLLMProvider:
        return CachedLLMProvider(OpenAICompatibleLLMProvider(
            api_key=api_key, base_url=base_url, model=model_name,
            timeout_seconds=60.0, max_retries=3,
        ), model_name=model_name, enabled=False)

    cross_llm = provider(model)
    primary_llm = provider(primary_model) if primary_model else None
    cross_judge = AnswerJudge(cross_llm)
    primary_judge = AnswerJudge(primary_llm) if primary_llm else None
    sem = asyncio.Semaphore(max(1, concurrency))

    async def score(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            arguments = {
                "query": row["query"], "gold_answer": row["gold_answer"],
                "predicted_answer": row["predicted_answer"],
                "expected_action": row.get("expected_action", "answer"),
                "predicted_action": row.get("predicted_action", "answer"),
                **row["judge_context"],
            }
            primary = await primary_judge.evaluate(**arguments) if primary_judge else None
            cross = await cross_judge.evaluate(**arguments)
            return {
                "id": row["id"], "turn_idx": row.get("turn_idx", 0),
                "evaluation_scope": row.get("evaluation_scope", "benchmark"),
                "primary_verdict": primary.verdict if primary else row.get("primary_judge_verdict"),
                "primary_score": primary.score if primary else row.get("primary_judge_score"),
                "primary_rationale": primary.rationale if primary else row.get("primary_judge_rationale"),
                "primary_parse_fallback": primary.parse_fallback if primary else row.get("primary_judge_parse_fallback"),
                "primary_comparable": primary is not None or row["primary_comparable"],
                "cross_verdict": cross.verdict, "cross_score": cross.score,
                "cross_rationale": cross.rationale, "parse_fallback": cross.parse_fallback,
                "judge_context": row["judge_context"],
                "reference_annotation_sha256": row["reference_annotation_sha256"],
            }

    scored = await asyncio.gather(*(score(row) for row in predictions))
    cross_valid = [row for row in scored if not row["parse_fallback"]]
    primary_valid = [row for row in scored if row["primary_comparable"] and row["primary_parse_fallback"] is False]
    pairs = [row for row in primary_valid if not row["parse_fallback"]]

    def weighted(records: list[dict[str, Any]], field: str) -> float | None:
        return round(sum(row[field] for row in records) / len(records) * 100, 2) if records else None

    return {
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "cross_judge_model": model, "primary_rejudge_model": primary_model,
        "attempted_turns": len(scored), "evaluated_turns": len(cross_valid),
        "parse_fallback_count": len(scored) - len(cross_valid),
        "primary_evaluated_turns": len(primary_valid),
        "primary_parse_fallback_count": sum(row["primary_parse_fallback"] is True for row in scored),
        "primary_unknown_parse_status_count": sum(row["primary_parse_fallback"] is None for row in scored),
        "comparable_pair_count": len(pairs),
        "primary_comparison_context_aligned": bool(scored) and all(row["primary_comparable"] for row in scored),
        "comparison_note": "Agreement uses only valid grades with the same context and annotation protocol. Historical primary grades without matching provenance are excluded; use --primary-model to rejudge both models.",
        "primary_weighted_accuracy": weighted(primary_valid, "primary_score"),
        "cross_weighted_accuracy": weighted(cross_valid, "cross_score"),
        "cross_strict_accuracy": round(sum(row["cross_verdict"] == "correct" for row in cross_valid) / len(cross_valid) * 100, 2) if cross_valid else None,
        "exact_verdict_agreement_pct": round(sum(row["primary_verdict"] == row["cross_verdict"] for row in pairs) / len(pairs) * 100, 2) if pairs else None,
        "within_half_score_agreement_pct": round(sum(abs(row["primary_score"] - row["cross_score"]) <= 0.5 for row in pairs) / len(pairs) * 100, 2) if pairs else None,
        "by_scope": {
            scope: {
                "attempted": sum(row["evaluation_scope"] == scope for row in scored),
                "evaluated": sum(row["evaluation_scope"] == scope for row in cross_valid),
                "weighted_accuracy": weighted([row for row in cross_valid if row["evaluation_scope"] == scope], "cross_score"),
            } for scope in sorted({row["evaluation_scope"] for row in scored})
        },
        "llm_usage": cross_llm.usage_snapshot(),
        "primary_llm_usage": primary_llm.usage_snapshot() if primary_llm else None,
        "disagreements": [row for row in pairs if row["primary_verdict"] != row["cross_verdict"]],
        "results": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "eval/reports/predictions_latest.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/dataset/jd_eval_benchmark.jsonl")
    parser.add_argument("--guardrail-dataset", type=Path, default=ROOT / "eval/dataset/jd_eval_guardrail_and_correction.jsonl")
    parser.add_argument("--knowledge-dir", type=Path, default=None, help="Policy corpus for judge references; defaults to Settings.knowledge_dir")
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--primary-model", default=None, help="Also rejudge the primary model on identical saved answers and references")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True, help="New output path; never overwrites a historical report")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.out}")
    prediction_bytes = args.predictions.read_bytes()
    reference_bytes = args.dataset.read_bytes()
    guard_bytes = args.guardrail_dataset.read_bytes() if args.guardrail_dataset.exists() else None
    decode = lambda data: [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    dataset = decode(reference_bytes) + (decode(guard_bytes) if guard_bytes else [])
    settings = Settings()
    reference_corpus = load_reference_corpus(args.knowledge_dir or settings.knowledge_dir)
    rows = hydrate_prediction_context(decode(prediction_bytes), dataset, reference_corpus=reference_corpus)
    report = asyncio.run(cross_judge_predictions(
        rows, api_key=args.api_key or settings.active_judge_api_key,
        base_url=args.base_url or settings.active_judge_base_url,
        model=args.model, primary_model=args.primary_model, concurrency=args.concurrency,
    ))
    report.update({
        "reference_dataset_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference_guardrail_sha256": hashlib.sha256(guard_bytes).hexdigest() if guard_bytes is not None else None,
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "reference_dataset": str(args.dataset.resolve()), "predictions_file": str(args.predictions.resolve()),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key not in {"results", "disagreements"}}, ensure_ascii=False, indent=2))
    print(f"Saved replay report to {args.out}")


if __name__ == "__main__":
    main()
