"""Reference-only context for grading; never pass these annotations to generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

JUDGE_PROTOCOL_VERSION = 3


def load_reference_corpus(knowledge_dir: Path | str) -> list[dict[str, Any]]:
    """Read every policy document once, independently of predicted answers/retrieval.

    Keep original version/event text, including historical policies. The judge must
    check applicability rather than assume every document applies to every question.
    """
    root = Path(knowledge_dir).resolve(strict=True)
    files = sorted(root.glob("*.md"))
    if not files:
        raise ValueError(f"No Markdown reference documents: {root}")
    documents = []
    for path in files:
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Reference document escapes knowledge directory: {path}")
        text = path.read_text(encoding="utf-8")
        documents.append({
            "source": "knowledge_corpus_full_document",
            "doc_id": path.stem,
            "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "quote": text,
        })
    return documents


def build_judge_context(
    item: dict[str, Any], turn: dict[str, Any], history: list[dict[str, str]],
    *, reference_corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = str(turn.get("expected_action") or item.get("expected_action") or "answer")
    evidence = turn.get("gold_evidence_refs") or [
        {"quote": quote} for quote in turn.get("gold_evidence_spans", [])
    ]
    return {
        "conversation_history": [dict(message) for message in history],
        "policy_context": dict(turn.get("policy_context") or item.get("policy_context") or {}),
        "reference_evidence": [*evidence, *(reference_corpus or [])],
        "allowed_actions": list(turn.get("allowed_actions") or [expected]),
        "answer_contract": turn.get("answer_contract") or item.get("answer_contract") or "",
        "clarification_contract": turn.get("clarification_contract") or item.get("clarification_contract") or {},
    }


def reference_sha256(turn: dict[str, Any]) -> str:
    annotations = {key: turn.get(key) for key in (
        "gold_answer", "expected_action", "allowed_actions", "gold_evidence_refs",
        "gold_evidence_spans", "answer_contract", "clarification_contract",
    )}
    return hashlib.sha256(json.dumps(annotations, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def hydrate_prediction_context(
    predictions: list[dict[str, Any]], dataset: list[dict[str, Any]],
    *, reference_corpus: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct prior ACTUAL turns, validating row identity against reference cases.

    The selected dataset supplies the reference annotations. Saved primary grades remain
    historical; callers must mark legacy/revised comparisons as non-comparable.
    """
    cases = {item["id"]: item for item in dataset}
    if len(cases) != len(dataset):
        raise ValueError("Duplicate reference case IDs")
    saved: dict[tuple[str, int], dict[str, Any]] = {}
    for row in predictions:
        index = row.get("turn_idx", 0)
        if type(index) is not int or index < 0:
            raise ValueError("turn_idx must be a nonnegative integer")
        key = (row["id"], index)
        if key in saved:
            raise ValueError(f"Duplicate prediction: {key}")
        saved[key] = row
    output = []
    for row in predictions:
        case_id, index = row["id"], row.get("turn_idx", 0)
        if case_id not in cases:
            raise ValueError(f"No reference case for {case_id}")
        item = cases[case_id]
        turns = [item, *item.get("follow_up_turns", [])]
        if index >= len(turns) or row["query"] != turns[index]["query"]:
            raise ValueError(f"Prediction/reference turn mismatch: {case_id}/{index}")
        history = []
        for prior in range(index):
            previous = saved.get((case_id, prior))
            if previous is None or previous["query"] != turns[prior]["query"]:
                raise ValueError(f"Missing/mismatched prior actual turn: {case_id}/{prior}")
            history.extend([
                {"role": "user", "content": previous["query"]},
                {"role": "assistant", "content": previous["predicted_answer"]},
            ])
        turn = turns[index]
        context = build_judge_context(item, turn, history, reference_corpus=reference_corpus)
        digest = reference_sha256(turn)
        output.append({
            **row,
            "gold_answer": turn.get("gold_answer", ""),
            "expected_action": turn.get("expected_action", "answer"),
            "judge_context": context,
            "reference_annotation_sha256": digest,
            "primary_comparable": (
                row.get("judge_protocol_version") == JUDGE_PROTOCOL_VERSION
                and row.get("reference_annotation_sha256") == digest
                and row.get("judge_context") == context
                and row.get("primary_judge_parse_fallback") is False
            ),
        })
    return output
