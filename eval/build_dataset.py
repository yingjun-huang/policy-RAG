"""Builds, resolves, lints, and merges sharded JD policy evaluation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.validate_assets import check_mutation_detection, validate_assets

BENCHMARK_SHARDS: tuple[tuple[str, str], ...] = (
    ("01_single_hop.jsonl", "single_hop"),
    ("02_multi_hop.jsonl", "multi_hop"),
    ("03_table_lookup.jsonl", "table_lookup"),
    ("04_version_contrast.jsonl", "version_contrast"),
    ("05_clarification_needed.jsonl", "clarification_needed"),
)
OPTIONAL_TEMPORAL_SHARD = ("04b_implicit_temporal.jsonl", "implicit_temporal")
GUARDRAIL_SHARD = "06_guardrail_and_correction.jsonl"

DEFAULT_POLICY_CONTEXT: dict[str, str] = {
    "namespace": "jd_policy",
    "corpus_revision": "jd-policy-v2",
    "as_of": "2024-06-01T12:00:00+08:00",
}
DEFAULT_ANSWER_CONTRACT = (
    "Return only conclusions entailed by the quoted policy and given facts; "
    "conditional claims must remain conditional."
)
DEFAULT_EVAL_INPUT_CONTRACT = (
    "query, prior actual conversation and policy_context are inputs; "
    "gold answers/evidence/known_slots/calculations are reference annotations, "
    "never retrieval or model inputs."
)

ANTI_LEAKAGE_PAT = re.compile(
    r"本库|库内|仿真|模拟|测试假设|模型|非免责|同价装箱|计费毛重|普通中小件|"
    r"送修0天|假一赔十标识|非保税仓备货|双方同意改折价退|普通快递中小件|"
    r"（(?:超过|大于|小于|低于|达到|不足)\s*\d+"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads non-empty JSON lines from path."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes normalized JSON lines with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def load_policy_texts(knowledge_dir: Path) -> dict[str, str]:
    """Loads active Markdown policy documents keyed by document stem."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(knowledge_dir.glob("*.md"))}


def _section_bounds(text: str, section: str) -> tuple[int, int]:
    """Returns (start, end) character offsets of `## {section}` block inside Markdown."""
    header = f"## {section}\n"
    pos = text.find(header)
    if pos == -1:
        raise ValueError(f"Section '## {section}' not found")
    nxt = text.find("\n## ", pos + len(header))
    return pos, (len(text) if nxt == -1 else nxt)


def resolve_evidence_ref(
    texts: dict[str, str],
    doc_id: str,
    section: str,
    quote: str,
    *,
    revision_id: str = "jd-policy-v2",
    evidence_id: str = "e1",
) -> dict[str, Any]:
    """Resolves exact character/line offsets and sha256 for a quoted policy span."""
    if doc_id not in texts:
        raise ValueError(f"Unknown policy document: {doc_id}")
    text = texts[doc_id]
    sec_start, sec_end = _section_bounds(text, section)
    local_idx = text.find(quote, sec_start, sec_end)
    if local_idx == -1:
        raise ValueError(f"Quote not found inside section '## {section}' of {doc_id}: {quote[:40]!r}")
    char_start = local_idx
    char_end = local_idx + len(quote)
    return {
        "evidence_id": evidence_id,
        "doc_id": doc_id,
        "revision_id": revision_id,
        "section": section,
        "quote": quote,
        "char_start": char_start,
        "char_end": char_end,
        "line_start": text[:char_start].count("\n") + 1,
        "line_end": text[:char_end].count("\n") + 1,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _normalize_turn(
    turn: dict[str, Any],
    texts: dict[str, str],
    *,
    default_policy_context: dict[str, Any],
) -> dict[str, Any]:
    """Normalizes either a top-level benchmark item or one of its follow-up turns."""
    raw_refs = turn.get("gold_evidence_refs") or []
    if not raw_refs:
        raise ValueError(f"{turn.get('id', 'turn')}: missing gold_evidence_refs")

    resolved_refs: list[dict[str, Any]] = []
    spans: list[str] = []
    ordered_docs: list[str] = []
    for idx, ref in enumerate(raw_refs, 1):
        resolved = resolve_evidence_ref(
            texts,
            ref["doc_id"],
            ref["section"],
            ref["quote"],
            revision_id=ref.get("revision_id", "jd-policy-v2"),
            evidence_id=ref.get("evidence_id", f"e{idx}"),
        )
        resolved_refs.append(resolved)
        spans.append(resolved["quote"])
        if resolved["doc_id"] not in ordered_docs:
            ordered_docs.append(resolved["doc_id"])

    if turn.get("gold_doc_ids"):
        if set(turn["gold_doc_ids"]) != set(ordered_docs):
            raise ValueError(
                f"{turn.get('id', 'turn')}: gold_doc_ids {turn['gold_doc_ids']} "
                f"differs from evidence docs {ordered_docs}"
            )
        doc_ids = list(turn["gold_doc_ids"])
    else:
        doc_ids = ordered_docs

    out = dict(turn)
    out["gold_doc_ids"] = doc_ids
    out["relevant_doc_ids"] = list(doc_ids)
    out["gold_evidence_spans"] = spans
    out["gold_evidence_refs"] = resolved_refs
    out.setdefault("policy_context", dict(default_policy_context))
    out.setdefault("known_slots", {})
    out.setdefault("missing_slots", [])
    out.setdefault("expected_action", "clarify" if out["missing_slots"] else "answer")
    return out


def normalize_benchmark_entry(
    entry: dict[str, Any],
    texts: dict[str, str],
    *,
    expected_category: str | None = None,
) -> dict[str, Any]:
    """Normalizes a benchmark entry and resolves all evidence references."""
    if expected_category and entry.get("category") not in (None, expected_category):
        raise ValueError(
            f"{entry.get('id')}: category {entry.get('category')!r} != shard category {expected_category!r}"
        )
    row = dict(entry)
    if expected_category and not row.get("category"):
        row["category"] = expected_category
    row.setdefault("policy_context", dict(DEFAULT_POLICY_CONTEXT))
    row = _normalize_turn(row, texts, default_policy_context=row["policy_context"])
    row.setdefault("answer_contract", DEFAULT_ANSWER_CONTRACT)
    row.setdefault("turn_index", 0)
    row.setdefault("evaluation_input_contract", DEFAULT_EVAL_INPUT_CONTRACT)

    if row.get("follow_up_turns"):
        row["follow_up_turns"] = [
            _normalize_turn(t, texts, default_policy_context=row["policy_context"])
            for t in row["follow_up_turns"]
        ]
    return row


def normalize_guardrail_entry(entry: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    """Validates and normalizes a guardrail/abstain/correction entry."""
    row = dict(entry)
    row.setdefault("policy_context", dict(DEFAULT_POLICY_CONTEXT))
    for turn in [row, *row.get("follow_up_turns", [])]:
        doc_ids = list(turn.get("gold_doc_ids") or [])
        turn["gold_doc_ids"] = doc_ids
        turn["relevant_doc_ids"] = list(doc_ids)
        for doc_id in doc_ids:
            if doc_id not in texts:
                raise ValueError(f"{row.get('id')}: unknown doc_id {doc_id}")
        for span in turn.get("gold_evidence_spans") or []:
            if not any(span in texts[d] for d in doc_ids):
                raise ValueError(f"{row.get('id')}: guardrail span not found in {doc_ids}: {span[:40]!r}")
    return row


def _char_bigrams(text: str) -> set[str]:
    clean = re.sub(r"[\W_]+", "", text)
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def lint_datasets(
    bench_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Checks queries for leakage terms, duplicates, and same-category near-duplicates."""
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()

    for label, rows in [("benchmark", bench_rows), ("guardrail", guard_rows)]:
        for row in rows:
            rid = row["id"]
            if rid in seen_ids:
                errors.append(f"Duplicate id: {rid}")
            seen_ids.add(rid)

            for idx, q in enumerate([row["query"], *[t["query"] for t in row.get("follow_up_turns", [])]]):
                tag = rid if idx == 0 else f"{rid}#t{idx}"
                if idx == 0:
                    if q in seen_queries:
                        errors.append(f"Duplicate top-level query in {label}: {tag}")
                    seen_queries.add(q)
                hit = ANTI_LEAKAGE_PAT.search(q)
                if hit:
                    errors.append(f"Anti-leakage pattern {hit.group(0)!r} in {tag}: {q}")

    by_cat: dict[str, list[tuple[str, set[str]]]] = {}
    for row in bench_rows + guard_rows:
        by_cat.setdefault(row["category"], []).append((row["id"], _char_bigrams(row["query"])))
    for cat, items in by_cat.items():
        for i in range(len(items)):
            id_a, bg_a = items[i]
            if not bg_a:
                continue
            for j in range(i + 1, len(items)):
                id_b, bg_b = items[j]
                if not bg_b:
                    continue
                jaccard = len(bg_a & bg_b) / len(bg_a | bg_b)
                if jaccard >= 0.80:
                    warnings.append(f"Near-duplicate queries in {cat}: {id_a} vs {id_b} (jaccard={jaccard:.2f})")

    return errors, warnings


def split_existing_to_shards(root: Path) -> dict[str, int]:
    """One-time lossless export from merged dataset files into category shards."""
    dataset_dir = root / "eval" / "dataset"
    shards_dir = dataset_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    bench = load_jsonl(dataset_dir / "jd_eval_benchmark.jsonl")
    guard = load_jsonl(dataset_dir / "jd_eval_guardrail_and_correction.jsonl")

    counts: dict[str, int] = {}
    for shard_name, category in BENCHMARK_SHARDS:
        subset = [r for r in bench if r["category"] == category]
        write_jsonl(shards_dir / shard_name, subset)
        counts[shard_name] = len(subset)

    write_jsonl(shards_dir / GUARDRAIL_SHARD, guard)
    counts[GUARDRAIL_SHARD] = len(guard)
    return counts


def build_datasets(
    root: Path,
    *,
    include_implicit: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    """Resolves, validates, and merges all shard files into final evaluation JSONL artifacts."""
    texts = load_policy_texts(root / "knowledge_jd")
    dataset_dir = root / "eval" / "dataset"
    shards_dir = dataset_dir / "shards"

    bench_rows: list[dict[str, Any]] = []
    shard_counts: dict[str, int] = {}

    active_shards = list(BENCHMARK_SHARDS)
    if include_implicit and (shards_dir / OPTIONAL_TEMPORAL_SHARD[0]).exists():
        active_shards.append(OPTIONAL_TEMPORAL_SHARD)

    for shard_name, category in active_shards:
        shard_path = shards_dir / shard_name
        raw_rows = load_jsonl(shard_path)
        norm_rows = [normalize_benchmark_entry(r, texts, expected_category=category) for r in raw_rows]
        bench_rows.extend(norm_rows)
        shard_counts[shard_name] = len(norm_rows)

    guard_raw = load_jsonl(shards_dir / GUARDRAIL_SHARD)
    guard_rows = [normalize_guardrail_entry(r, texts) for r in guard_raw]
    shard_counts[GUARDRAIL_SHARD] = len(guard_rows)

    lint_errors, warnings = lint_datasets(bench_rows, guard_rows)
    report = validate_assets(texts, bench_rows, guard_rows)
    if lint_errors:
        report["errors"].extend(lint_errors)
        report["status"] = "FAIL"

    if report["status"] == "PASS":
        muts = check_mutation_detection(texts, bench_rows)
        report["mutation_detection"] = muts
        if not all(m["detected"] for m in muts):
            report["errors"].append("mutation detection probe failed")
            report["status"] = "FAIL"

    bench_out = dataset_dir / "jd_eval_benchmark.jsonl"
    guard_out = dataset_dir / "jd_eval_guardrail_and_correction.jsonl"

    if check_only:
        cur_bench = load_jsonl(bench_out) if bench_out.exists() else []
        cur_guard = load_jsonl(guard_out) if guard_out.exists() else []
        if cur_bench != bench_rows:
            report["errors"].append("jd_eval_benchmark.jsonl is out of sync with shards (run python -m eval.build_dataset)")
            report["status"] = "FAIL"
        if cur_guard != guard_rows:
            report["errors"].append(
                "jd_eval_guardrail_and_correction.jsonl is out of sync with shards (run python -m eval.build_dataset)"
            )
            report["status"] = "FAIL"
    elif report["status"] == "PASS":
        write_jsonl(bench_out, bench_rows)
        write_jsonl(guard_out, guard_rows)

    doc_hits = Counter(d for r in bench_rows for d in r["gold_doc_ids"])
    report["shard_counts"] = shard_counts
    report["doc_coverage"] = {
        "covered_docs": len(doc_hits),
        "min_hits": min(doc_hits.values()) if doc_hits else 0,
        "max_hits": max(doc_hits.values()) if doc_hits else 0,
    }
    report["warnings"] = warnings
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate merged JD policy evaluation datasets")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split-existing", action="store_true", help="Export existing merged datasets into shards")
    parser.add_argument("--check", action="store_true", help="Verify shards validate and match merged files without writing")
    parser.add_argument("--include-implicit", action="store_true", help="Include optional 04b_implicit_temporal.jsonl")
    args = parser.parse_args()

    if args.split_existing:
        exported = split_existing_to_shards(args.root)
        print(json.dumps({"action": "split_existing", "shard_counts": exported}, ensure_ascii=False, indent=2))
        return

    report = build_datasets(args.root, include_implicit=args.include_implicit, check_only=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
