"""F5 Benchmark: Re-evaluate saved predictions using an independent Judge LLM to audit self-judge bias."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judges import AnswerJudge
from policy_rag.config import Settings
from policy_rag.providers.llm_cache import CachedLLMProvider
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider


async def cross_judge_predictions(
    predictions: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    concurrency: int = 6,
) -> dict[str, Any]:
    raw_llm = OpenAICompatibleLLMProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=60.0,
        max_retries=3,
    )
    cached_llm = CachedLLMProvider(
        raw_llm,
        db_path=".cache/llm/cache.sqlite",
        model_name=model,
        enabled=True,
    )
    judge = AnswerJudge(cached_llm)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _score(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            j = await judge.evaluate(
                query=row["query"],
                gold_answer=row["gold_answer"],
                predicted_answer=row["predicted_answer"],
                expected_action=row.get("expected_action", "answer"),
                predicted_action=row.get("predicted_action", "answer"),
            )
            return {
                "id": row.get("id"),
                "turn_idx": row.get("turn_idx", 0),
                "primary_verdict": row.get("primary_judge_verdict"),
                "primary_score": float(row.get("primary_judge_score", 0.0)),
                "cross_verdict": j.verdict,
                "cross_score": j.score,
                "cross_rationale": j.rationale,
                "parse_fallback": j.parse_fallback,
            }

    scored = await asyncio.gather(*[_score(r) for r in predictions])

    n = len(scored)
    exact_agree = sum(1 for r in scored if r["primary_verdict"] == r["cross_verdict"])
    close_agree = sum(1 for r in scored if abs(r["primary_score"] - r["cross_score"]) <= 0.5)
    parse_fallbacks = sum(1 for r in scored if r.get("parse_fallback"))
    primary_weighted = sum(r["primary_score"] for r in scored) / n * 100 if n else 0.0
    cross_weighted = sum(r["cross_score"] for r in scored) / n * 100 if n else 0.0
    cross_strict = sum(1 for r in scored if r["cross_verdict"] == "correct") / n * 100 if n else 0.0

    cached_llm.flush_ledger(f"scripts.cross_model_judge.{model}")

    return {
        "cross_judge_model": model,
        "evaluated_turns": n,
        "parse_fallback_count": parse_fallbacks,
        "primary_weighted_accuracy": round(primary_weighted, 2),
        "cross_weighted_accuracy": round(cross_weighted, 2),
        "cross_strict_accuracy": round(cross_strict, 2),
        "exact_verdict_agreement_pct": round(exact_agree / n * 100, 2) if n else 0.0,
        "within_half_score_agreement_pct": round(close_agree / n * 100, 2) if n else 0.0,
        "llm_usage": cached_llm.usage_snapshot(),
        "disagreements": [r for r in scored if r["primary_verdict"] != r["cross_verdict"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model LLM Judge audit on saved predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(ROOT / "eval" / "reports" / "predictions_latest.jsonl"),
    )
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "eval" / "reports" / "cross_judge_report.json"),
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    rows = [json.loads(ln) for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    settings = Settings()
    api_key = args.api_key or settings.active_judge_api_key
    base_url = args.base_url or settings.active_judge_base_url

    report = asyncio.run(
        cross_judge_predictions(
            rows,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            concurrency=args.concurrency,
        )
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "disagreements"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved cross-judge audit to {out_path}")


if __name__ == "__main__":
    main()
