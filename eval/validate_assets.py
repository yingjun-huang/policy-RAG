"""Standalone integrity & consistency validator for Scheme-A JD policy KB and eval datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from policy_rag.indexing.chunking import chunk_markdown, parse_version_banner

MIN_CATEGORY_COUNTS = {
    "single_hop": 25,
    "multi_hop": 25,
    "table_lookup": 20,
    "version_contrast": 15,
    "clarification_needed": 15,
}
ALLOWED_CATEGORIES = set(MIN_CATEGORY_COUNTS) | {"implicit_temporal"}

VERSION_CHAINS: dict[str, tuple[str, str]] = {
    "shipping": ("jd_ver_01_shipping_2023_legacy", "jd_ship_01_freight_standard"),
    "fresh_after_sales": ("jd_ver_02_fresh_2024_legacy", "jd_srv_01_fresh_instant_pay"),
}

_FORBIDDEN_BODY_PAT = re.compile(
    r"仿真|模拟|本库|模型|测试假设|人为设置|人为构造|不代表京东|不是京东|不证明|不声称|指导现实|法定保底|海关承诺|国家标准|工程评测|样本"
)


def _dec(val: Any) -> Decimal:
    return Decimal(str(val))


def _cents(val: Any) -> Decimal:
    return _dec(val).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _ceil(val: Any) -> Decimal:
    return _dec(val).to_integral_value(rounding=ROUND_CEILING)


def _extract_table(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return [
        [cell.strip() for cell in ln.strip("|").split("|")]
        for ln in lines
        if not re.fullmatch(r"[\s|:-]+", ln)
    ]


def _extract_banner_meta(doc_id: str, text: str) -> dict[str, Any]:
    lines = text.splitlines()
    h1_indices = [i for i, ln in enumerate(lines) if ln.startswith("# ")]
    if len(h1_indices) != 1:
        return {}
    for ln in lines[h1_indices[0] + 1 : h1_indices[0] + 6]:
        if ln.strip().startswith("政策版本："):
            meta = parse_version_banner(ln)
            meta["doc_id"] = doc_id
            return meta
    return {}


def _to_shanghai_naive(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
    return dt


def validate_assets(
    texts: dict[str, str],
    cases: list[dict[str, Any]],
    guardrail_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validates Scheme-A Markdown documents, chunk metadata extraction, benchmark references, and calculations."""
    errors: list[str] = []
    counts: Counter[str] = Counter()
    meta: dict[str, dict[str, Any]] = {}

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. Document format, Scheme-A header banner, tables & chunker extraction
    for doc_id, text in sorted(texts.items()):
        check(not text.lstrip().startswith("---"), f"{doc_id}: must not contain YAML frontmatter")
        check(len(re.findall(r"^# ", text, re.M)) == 1, f"{doc_id}: must have exactly 1 H1")
        check(not _FORBIDDEN_BODY_PAT.search(text), f"{doc_id}: leftover defensive simulation wording")
        m = _extract_banner_meta(doc_id, text)
        meta[doc_id] = m
        check(bool(m.get("version")) and bool(m.get("effective_from")), f"{doc_id}: missing/invalid 政策版本 banner")
        check(m.get("applicability_event") is not None, f"{doc_id}: missing 适用事件 in version banner")
        if m.get("effective_from"):
            start = datetime.fromisoformat(m["effective_from"])
            end = datetime.fromisoformat(m["effective_to"]) if m.get("effective_to") else None
            check(end is None or start < end, f"{doc_id}: invalid effective date interval")
            expected_status = "archived" if end is not None else "current"
            check(m.get("document_status") == expected_status, f"{doc_id}: banner status mismatch")

        check(sum(ln.strip().startswith("```") for ln in text.splitlines()) % 2 == 0, f"{doc_id}: unbalanced code fence")
        widths = [len(ln.strip("|").split("|")) for ln in text.splitlines() if ln.startswith("|")]
        check(len(set(widths)) <= 1, f"{doc_id}: inconsistent table column counts")
        MarkdownIt("commonmark").enable("table").parse(text)

        chunks = chunk_markdown(text, document_id=doc_id, doc_slug=doc_id)
        check(bool(chunks), f"{doc_id}: produced 0 chunks")
        check(all(c.section != "概述" for c in chunks), f"{doc_id}: unexpected stub '概述' chunk")
        check(
            all(
                c.metadata.get("effective_from") == m.get("effective_from")
                and c.metadata.get("document_status") == m.get("document_status")
                for c in chunks
            ),
            f"{doc_id}: chunker failed to auto-extract banner metadata",
        )

    check(len(texts) == 32, "document count must be 32")
    check(len(cases) >= 100, "top-level case count must be >= 100")
    check(len({x["id"] for x in cases}) == len(cases), "duplicate case IDs")
    check(len({x["query"] for x in cases}) == len(cases), "duplicate queries")
    cat_counts = dict(Counter(x["category"] for x in cases))
    unexpected_cats = set(cat_counts) - ALLOWED_CATEGORIES
    check(not unexpected_cats, f"unexpected categories: {sorted(unexpected_cats)}")
    for cat, min_count in MIN_CATEGORY_COUNTS.items():
        check(cat_counts.get(cat, 0) >= min_count, f"category '{cat}' below minimum {min_count}: {cat_counts.get(cat, 0)}")
    check(set(texts) == {d for r in cases for d in r["gold_doc_ids"]}, "uncovered document in benchmark")

    # 2. Version chain continuity probes
    for family, (old_id, new_id) in VERSION_CHAINS.items():
        old_m, new_m = meta.get(old_id, {}), meta.get(new_id, {})
        check(old_m.get("effective_to") == new_m.get("effective_from"), f"{family}: version cutover gap")
        check(
            old_m.get("applicability_event") == new_m.get("applicability_event"),
            f"{family}: inconsistent applicability_event across versions",
        )

    # 3. Rule-derived independent calculation recomputer
    large_rows = {r[0]: r for r in _extract_table(texts["jd_cat_03_appliances_large"])[1:]}
    accessory_rows = {r[0]: r for r in _extract_table(texts["jd_cat_01_mobile_phones"])[1:]}
    remote_rows = {r[0]: r for r in _extract_table(texts["jd_ship_02_remote_freight"])[1:]}
    ship = texts["jd_ship_01_freight_standard"]
    threshold = _dec(re.search(r"满(\d+)元免基础运费", ship).group(1))
    base_fee = _dec(re.search(r"基础运费(\d+)元", ship).group(1))
    weight_limit = _dec(re.search(r"选择(\d+)kg首重", ship).group(1))
    weight_rate = _dec(re.search(r"×(\d+)元/kg", ship).group(1))
    remote_limit = _dec(re.search(r"首重统一(\d+)kg", texts["jd_ship_02_remote_freight"]).group(1))
    fruit_rows = _extract_table(texts["jd_cat_05_fresh_fruits"])[1:]
    check(
        [r[0] for r in fruit_rows] == ["0<r<10%", "10%≤r<30%", "30%≤r<50%", "50%≤r≤100%"],
        "fruit intervals must partition (0, 100%] cleanly",
    )

    def calc_shipping(c: dict[str, Any]) -> dict[str, Decimal]:
        amount, weight = _dec(c["amount"]), _dec(c["weight"])
        base = Decimal(0) if c["plus"] or amount >= threshold else base_fee
        extra = _ceil(max(weight - weight_limit, Decimal(0))) * weight_rate
        region = c["region"]
        remote = Decimal(0)
        if region != "正常地区":
            row = remote_rows[region]
            fixed = _dec(re.search(r"(\d+)元", row[1]).group(1))
            rate = _dec(re.search(r"(\d+)元/kg", row[2]).group(1))
            remote = fixed + _ceil(max(weight - remote_limit, Decimal(0))) * rate
        return {
            "base": _cents(base),
            "regular_extra": _cents(extra),
            "remote_extra": _cents(remote),
            "total": _cents(base + extra + remote),
        }

    def recompute(c: dict[str, Any]) -> dict[str, Any]:
        kind = c["kind"]
        p = _dec(c.get("amount", 0))
        if kind == "depreciation":
            row = large_rows[c["product"]]
            rate = _dec(row[1].rstrip("%")) / 100
            days = c["elapsed_days"] - c["repair_days"] - c["waiting_days"]
            check(days >= 0, "depreciation: repairs exceed elapsed days")
            return {
                "fee": _cents(min(_dec(c["invoice_price"]) * rate * days, _dec(c["invoice_price"]))),
                "rate": row[1],
                "major_warranty_years": int(row[3].removesuffix("年")),
            }
        if kind == "fruit":
            r = _dec(c["damaged_count"]) / _dec(c["total_count"])
            if r == 0:
                low = high = Decimal(0)
            elif r < Decimal("0.1"):
                low = high = p * r
            elif r < Decimal("0.3"):
                rates = re.findall(r"(\d+)%", fruit_rows[1][1])
                low, high = [p * _dec(v) / 100 for v in rates]
            elif r < Decimal("0.5"):
                rates = re.findall(r"(\d+)%", fruit_rows[2][1])
                low, high = [p * _dec(v) / 100 for v in rates]
            else:
                low = high = p
            return {"low": _cents(low), "high": _cents(high)}
        if kind == "live":
            count = c["total_count"]
            ratio = _dec(c["dead_count"]) / _dec(count)
            return {"refund": _cents(p if count > 2 and ratio >= Decimal("0.5") else p * ratio)}
        if kind == "shipping":
            return calc_shipping(c)
        if kind == "shipping_pair":
            return {"orders": [calc_shipping(o) for o in c["orders"]]}
        if kind == "shipping_and_pickup":
            rng = re.search(r"基础取件费(\d+)–(\d+)元", texts["jd_ship_05_pickup_service"])
            check(bool(rng), "missing pickup fee range in jd_ship_05")
            return {
                "orders": [calc_shipping(o) for o in c["orders"]],
                "pickup_normal_low": _cents(rng.group(1)),
                "pickup_normal_high": _cents(rng.group(2)),
                "pickup_plus": _cents(0),
            }
        if kind == "remote_only":
            row = remote_rows[c["region"]]
            fixed = _dec(re.search(r"(\d+)元", row[1]).group(1))
            rate = _dec(re.search(r"(\d+)元/kg", row[2]).group(1))
            return {"total": _cents(fixed + _ceil(max(_dec(c["weight"]) - remote_limit, Decimal(0))) * rate)}
        if kind == "accessory":
            keys = {
                "charger": "原装快充充电器",
                "cable": "原装数据线",
                "documents_group": "卡针/保修卡/说明书组合",
                "box": "销售包装盒破损",
                "film": "出厂屏幕保护膜撕除",
            }
            low = high = Decimal(0)
            for item in c["items"]:
                nums = [_dec(x) for x in re.findall(r"\d+(?:\.\d+)?", accessory_rows[keys[item]][1])]
                if "explicit_price" in c:
                    val = _dec(c["explicit_price"])
                    check(nums[0] <= val <= nums[-1], "accessory explicit price outside table")
                    lo = hi = val
                else:
                    lo, hi = nums[0], nums[-1]
                low += lo
                high += hi
            return {"low": _cents(low), "high": _cents(high)}
        if kind == "delay":
            return {"compensation": _cents(Decimal(5) if p <= 100 else min(max(p * Decimal("0.05"), 5), 50))}
        if kind == "stock":
            return {"compensation": _cents(min(max(p * Decimal("0.3"), 10), 500))}
        if kind == "fake":
            comp = max(p * 10, 1000)
            return {"refund": _cents(p), "compensation": _cents(comp), "total": _cents(p + comp)}
        if kind == "damaged_units":
            return {"refund": _cents(p * _dec(c["damaged_count"]) / _dec(c["total_count"]))}
        if kind == "insurance":
            return {"buyer_difference": _cents(max(_dec(c["actual"]) - _dec(c["limit"]), 0))}
        if kind == "quality_refund":
            return {"total": _cents(p + _dec(c["outbound"]))}
        if kind == "price_protection":
            pp_rows = _extract_table(texts["jd_srv_03_price_protection"])[1:]
            pp_map = {r[0]: int(r[1].removesuffix("天")) for r in pp_rows}
            default_days = pp_map[c["category_row"]]
            refund = max(
                _dec(c["paid"]) - _dec(c["current_price"]) - _dec(c.get("already_claimed", 0)),
                Decimal(0),
            )
            return {"default_days": default_days, "refund": _cents(refund)}
        if kind == "eggs":
            ratio = _dec(c["damaged_count"]) / _dec(c["total_count"])
            return {"refund": _cents(p if ratio > Decimal("0.3") else p * ratio)}
        if kind == "liquid_leak":
            ratio = _dec(c["leak_ratio"])
            return {"refund": _cents(p if ratio > Decimal("0.3") else p * ratio)}
        if kind == "fake_shipping":
            return {"compensation": _cents(min(max(p * Decimal("0.2"), 20), 200))}
        raise ValueError(f"unknown calculation kind: {kind}")

    def check_calculation(entry: dict[str, Any], label: str) -> None:
        c = entry.get("gold_calculation")
        if not c:
            return
        counts["calculation_entries"] += 1
        try:
            result = recompute(c)
            for key, value in result.items():
                if key == "orders":
                    for i, order in enumerate(value):
                        check(_cents(c["orders"][i]["total"]) == order["total"], f"{label}: order {i} total")
                        for subk in ("base", "regular_extra", "remote_extra"):
                            if subk in c["orders"][i]:
                                check(_cents(c["orders"][i][subk]) == order[subk], f"{label}: order {i} {subk}")
                elif isinstance(value, Decimal):
                    check(_cents(c[key]) == value, f"{label}: calc {key} expected {value}, got {c.get(key)}")
                else:
                    check(c[key] == value, f"{label}: calc {key}")
            if c["kind"] == "depreciation":
                known = entry["known_slots"]
                check(
                    known.get("days_since_receipt", 0) > 7
                    and known.get("exchange_eligible") is True
                    and known.get("has_same_model_stock") is True
                    and known.get("depreciated_return_agreed") is True,
                    f"{label}: depreciation eligibility",
                )
            ans_nums = {_dec(v) for v in re.findall(r"\d+(?:\.\d+)?", entry["gold_answer"])}
            for key, value in result.items():
                if key == "orders":
                    for order in value:
                        check(order["total"] in ans_nums, f"{label}: answer omits order total")
                elif isinstance(value, Decimal):
                    check(value in ans_nums, f"{label}: answer omits {key}={value}")
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as exc:
            errors.append(f"{label}: calculation error {exc}")

    def check_entry(entry: dict[str, Any], label: str) -> None:
        ids = entry["gold_doc_ids"]
        spans = entry["gold_evidence_spans"]
        refs = entry["gold_evidence_refs"]
        check(bool(ids) and bool(spans), f"{label}: empty evidence")
        check(len(spans) == len(refs), f"{label}: evidence/ref count mismatch")
        check(len(set(ids)) == len(ids), f"{label}: duplicate doc IDs")
        check(entry.get("relevant_doc_ids") == ids, f"{label}: relevant_doc_ids mismatch")
        covered: set[str] = set()
        for idx, (quote, ref) in enumerate(zip(spans, refs), 1):
            counts["evidence_spans"] += 1
            d = ref["doc_id"]
            check(d in ids and d in texts, f"{label}: unknown evidence doc {d}")
            if d not in texts:
                continue
            text = texts[d]
            covered.add(d)
            check(quote == ref["quote"] and text[ref["char_start"] : ref["char_end"]] == quote, f"{label}: exact quote/offset e{idx}")
            check(text[: ref["char_start"]].count("\n") + 1 == ref["line_start"], f"{label}: line_start e{idx}")
            check(text[: ref["char_end"]].count("\n") + 1 == ref["line_end"], f"{label}: line_end e{idx}")
            check(hashlib.sha256(text.encode("utf-8")).hexdigest() == ref["document_sha256"], f"{label}: sha256 e{idx}")
            check("## " + ref["section"] in text, f"{label}: missing section '{ref['section']}' in {d}")
        check(covered == set(ids), f"{label}: listed doc lacks evidence")

        known = entry.get("known_slots", {})
        missing = entry["missing_slots"]
        check(len(set(missing)) == len(missing), f"{label}: duplicate missing_slots")
        check(not (set(missing) & set(known)), f"{label}: asking already-known slot")
        check(entry["expected_action"] in ("answer", "clarify"), f"{label}: invalid expected_action")
        check((entry["expected_action"] == "clarify") == bool(missing), f"{label}: action/slots mismatch")
        if missing:
            qs = entry.get("clarification_questions", [])
            check({s for q in qs for s in q["slots"]} == set(missing), f"{label}: question/slot mismatch")
            check(all(q["question"] in entry["gold_answer"] for q in qs), f"{label}: question absent in answer")
        check_calculation(entry, label)

    # 4. Validate all top-level cases and follow-up turns
    for row in cases:
        check_entry(row, row["id"])
        counts["top_level_cases"] += 1
        if row["category"] == "multi_hop":
            check(len(row["gold_doc_ids"]) >= 2, f"{row['id']}: multi-hop requires >=2 docs")
        if row["category"] == "version_contrast":
            check(len(row["gold_doc_ids"]) >= 2, f"{row['id']}: version_contrast requires >=2 docs")
            check(bool(row.get("version_decisions")), f"{row['id']}: missing version_decisions")
            for vd in row.get("version_decisions", []):
                doc_id = vd["applicable_doc_id"]
                check(doc_id in VERSION_CHAINS.get(vd["policy_family"], ()), f"{row['id']}: unexpected family/doc pair")
                m = meta.get(doc_id, {})
                check(vd.get("event_type") == m.get("applicability_event"), f"{row['id']}: wrong/missing applicability event type")
                at = _to_shanghai_naive(vd["event_at"])
                start = datetime.fromisoformat(m["effective_from"])
                end = datetime.fromisoformat(m["effective_to"]) if m.get("effective_to") else None
                check(start <= at and (end is None or at < end), f"{row['id']}: event_at outside effective range")
        else:
            as_of_dt = _to_shanghai_naive(row["policy_context"]["as_of"])
            for d in row["gold_doc_ids"]:
                m = meta[d]
                start = datetime.fromisoformat(m["effective_from"])
                end = datetime.fromisoformat(m["effective_to"]) if m.get("effective_to") else None
                check(start <= as_of_dt and (end is None or as_of_dt < end), f"{row['id']}: cites non-active doc {d}")

        if row["category"] == "clarification_needed":
            follow = row.get("follow_up_turns", [])
            check(bool(follow), f"{row['id']}: missing follow_up_turns")
            prev = row["known_slots"]
            for idx, turn in enumerate(follow, 1):
                check_entry(turn, f"{row['id']}/turn{idx}")
                counts["follow_up_turns"] += 1
                check(all(turn["known_slots"].get(k) == v for k, v in prev.items()), f"{row['id']}/turn{idx}: lost confirmed slot")
                prev = turn["known_slots"]
            check(bool(follow) and follow[-1]["expected_action"] == "answer", f"{row['id']}: terminal turn not answered")

    # 5. Validate guardrail & correction cases when provided
    if guardrail_cases is not None:
        for row in guardrail_cases:
            counts["guardrail_cases"] += 1
            check(not _FORBIDDEN_BODY_PAT.search(row.get("query", "") + row.get("gold_answer", "")), f"{row['id']}: defensive wording")
            for turn in [row, *row.get("follow_up_turns", [])]:
                for doc_id in turn.get("gold_doc_ids", []):
                    check(doc_id in texts, f"{row['id']}: unknown doc_id {doc_id}")
                for sp in turn.get("gold_evidence_spans", []):
                    check(any(sp in texts[d] for d in turn.get("gold_doc_ids", []) if d in texts), f"{row['id']}: span not found verbatim")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": {
            "documents": len(texts),
            **dict(counts),
        },
        "categories": dict(Counter(x["category"] for x in cases)),
    }


def check_mutation_detection(texts: dict[str, str], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verifies that planted defects are caught by validate_assets rather than passing silently."""
    probes: list[dict[str, Any]] = []

    def run_probe(name: str, mutate: Any) -> None:
        d, c = dict(texts), copy.deepcopy(cases)
        mutate(d, c)
        try:
            detected = validate_assets(d, c)["status"] == "FAIL"
        except (ValueError, KeyError, TypeError, IndexError):
            detected = True
        probes.append({"name": name, "detected": detected})

    dep_idx = next(i for i, x in enumerate(cases) if (x.get("gold_calculation") or {}).get("kind") == "depreciation")
    ref_idx = next(i for i, x in enumerate(cases) if x.get("gold_evidence_refs"))
    vc_idx = next(
        i
        for i, x in enumerate(cases)
        if x.get("category") == "version_contrast"
        and any(vd.get("applicable_doc_id") == "jd_ship_01_freight_standard" for vd in x.get("version_decisions", []))
    )
    vc_sub = next(
        j
        for j, vd in enumerate(cases[vc_idx]["version_decisions"])
        if vd.get("applicable_doc_id") == "jd_ship_01_freight_standard"
    )

    run_probe("wrong_depreciation", lambda d, c: c[dep_idx]["gold_calculation"].update(fee="9999.99"))
    run_probe("altered_quote", lambda d, c: c[ref_idx]["gold_evidence_refs"][0].update(quote="not an original quotation"))
    run_probe(
        "wrong_version_at_cutover",
        lambda d, c: c[vc_idx]["version_decisions"][vc_sub].update(applicable_doc_id="jd_ver_01_shipping_2023_legacy"),
    )
    run_probe("yaml_reintroduced", lambda d, c: d.update(jd_cat_01_mobile_phones="---\nid: foo\n---\n" + d["jd_cat_01_mobile_phones"]))
    run_probe("changed_document", lambda d, c: d.update(jd_cat_01_mobile_phones=d["jd_cat_01_mobile_phones"].replace("99–199元", "90–190元")))
    return probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Scheme-A knowledge_jd and eval datasets")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((args.root / "knowledge_jd").glob("*.md"))}
    bench = [json.loads(s) for s in (args.root / "eval" / "dataset" / "jd_eval_benchmark.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    guard_path = args.root / "eval" / "dataset" / "jd_eval_guardrail_and_correction.jsonl"
    guard = [json.loads(s) for s in guard_path.read_text(encoding="utf-8").splitlines() if s.strip()] if guard_path.exists() else None

    report = validate_assets(texts, bench, guard)
    if report["status"] == "PASS":
        muts = check_mutation_detection(texts, bench)
        report["mutation_detection"] = muts
        if not all(m["detected"] for m in muts):
            report["errors"].append("mutation detection probe failed")
            report["status"] = "FAIL"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
