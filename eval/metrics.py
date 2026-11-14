"""Information retrieval and performance evaluation metrics."""

from __future__ import annotations


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Computes Precision@K = (relevant in top K) / K."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(item in relevant for item in top) / len(top)


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Computes Recall@K = (relevant in top K) / (total relevant)."""
    if not relevant:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def hit_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Computes Hit@K = 1.0 if at least one relevant document is in top K, else 0.0."""
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    """Computes Mean Reciprocal Rank = 1 / first_relevant_rank."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def complete_evidence_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Computes 1.0 iff ALL gold relevant docs appear within top K retrieved docs."""
    if not relevant:
        return 1.0
    return 1.0 if relevant.issubset(set(retrieved[:k])) else 0.0


def no_archived_leak_at_k(archived_docs: set[str], retrieved: list[str], k: int) -> float:
    """Computes 1.0 iff NO archived doc appears in top K retrieved docs."""
    return 0.0 if any(item in archived_docs for item in retrieved[:k]) else 1.0


def percentile(values: list[float], p: float) -> float:
    """Computes the p-th percentile of a list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return float(ordered[index])
