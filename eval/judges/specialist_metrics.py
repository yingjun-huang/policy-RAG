"""Keep action, retrieval, semantic grading and parse failures separate."""
from __future__ import annotations

from typing import Any


def summarize_judgments(records: list[dict[str, Any]], status: str) -> dict[str, Any]:
    judged = [record["judgment"] for record in records if record.get("judgment") is not None]
    valid = [judgment for judgment in judged if not judgment.parse_fallback]
    verdicts = {v: sum(j.verdict == v for j in valid) for v in ("correct", "partial", "incorrect")}
    return {
        "status": status if judged else "not_evaluated_missing_reference",
        "attempted": len(judged),
        "evaluated": len(valid),
        "missing_reference_count": len(records) - len(judged),
        "parse_fallback_count": len(judged) - len(valid),
        **verdicts,
        "strict_accuracy": round(verdicts["correct"] / len(valid) * 100, 2) if valid else None,
        "weighted_accuracy": round(sum(j.score for j in valid) / len(valid) * 100, 2) if valid else None,
    }
