"""Suggest candidate rule-graph edges from explicit cross-document citations in Markdown corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.retrieval.clause_graph import logical_doc_id_of

_BOOK_TITLE_RE = re.compile(r"《([^》\n]{2,40})》")
_CLAUSE_REF_RE = re.compile(r"(?:参照|依据|适用|按|见|详见)([^，。；\n]{2,30}(?:规则|总则|规范|办法|协议|标准|条款))")


def extract_candidate_edges(
    knowledge_dir: Path | str,
    *,
    config_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Scans Markdown corpus for explicit title/rule mentions and returns candidate edges."""
    chunks = load_and_chunk_knowledge(knowledge_dir)
    docs: dict[str, dict[str, str]] = {}
    for c in chunks:
        doc_id = logical_doc_id_of(c)
        entry = docs.setdefault(doc_id, {"title": c.title, "text": ""})
        entry["text"] += "\n" + c.content

    configured_pairs: set[tuple[str, str]] = set()
    if config_path and Path(config_path).exists():
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        doc_groups = {
            logical_doc_id_of(c): str((c.metadata or {}).get("doc_group") or "")
            for c in chunks
        }
        for edge in raw.get("edges", []):
            tgt = str(edge.get("tgt") or "")
            if edge.get("src"):
                configured_pairs.add((str(edge["src"]), tgt))
            elif edge.get("src_group"):
                grp = str(edge["src_group"])
                for did, dgrp in doc_groups.items():
                    if dgrp == grp and did != tgt:
                        configured_pairs.add((did, tgt))

    # Build title -> doc_id index
    title_index: list[tuple[str, str]] = []
    for doc_id, info in docs.items():
        clean_title = info["title"].strip()
        if clean_title:
            title_index.append((doc_id, clean_title))

    suggestions: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for src_id, info in sorted(docs.items()):
        text = info["text"]
        mentions: list[tuple[str, str]] = []
        for m in _BOOK_TITLE_RE.finditer(text):
            mentions.append(("book_title", m.group(1).strip()))
        for m in _CLAUSE_REF_RE.finditer(text):
            mentions.append(("clause_phrase", m.group(1).strip()))

        for pattern_kind, phrase in mentions:
            for tgt_id, tgt_title in title_index:
                if tgt_id == src_id:
                    continue
                if phrase in tgt_title or tgt_title in phrase:
                    pair = (src_id, tgt_id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    suggestions.append(
                        {
                            "src": src_id,
                            "tgt": tgt_id,
                            "tgt_title": tgt_title,
                            "pattern_kind": pattern_kind,
                            "matched_phrase": phrase,
                            "already_configured": pair in configured_pairs,
                        }
                    )

    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest policy graph edges from explicit citations")
    parser.add_argument("--knowledge-dir", type=str, default="knowledge_jd")
    parser.add_argument("--config", type=str, default="config/policy_graph.yaml")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    suggestions = extract_candidate_edges(args.knowledge_dir, config_path=args.config)
    payload = {
        "knowledge_dir": args.knowledge_dir,
        "config_path": args.config,
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
