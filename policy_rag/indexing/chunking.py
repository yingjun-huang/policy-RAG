"""Markdown chunking with frontmatter parsing, table preservation, and parent-child context."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from policy_rag.indexing.cleaning import normalize_text
from policy_rag.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
_BOILERPLATE_NOTICE_PREFIX = "> 本文为Policy-RAG工程评测的仿真规则"
_VERSION_LINE_PREFIX = "政策版本："
_VERSION_ID_RE = re.compile(r"^政策版本[：:]\s*([^\s|｜(（]+)")
_EFFECTIVE_FROM_RE = re.compile(r"生效期[：:]\s*(\d{4}-\d{2}-\d{2})")
_EFFECTIVE_TO_RE = re.compile(r"至\s*(\d{4}-\d{2}-\d{2})")
_APPLICABILITY_EVENT_RE = re.compile(r"适用事件[：:]\s*([^|｜\n]+)")
_EVENT_LABEL_TO_CODE: dict[str, str] = {
    "订单支付时间": "order_paid_at",
    "售后申请提交时间": "after_sales_requested_at",
    "退货/换货申请时间": "return_requested_at",
    "定制合同确认时间": "contract_confirmed_at",
    "纠纷申请时间": "dispute_requested_at",
    "退款申请时间": "refund_requested_at",
    "价保申请时间": "price_protection_requested_at",
}
_DOC_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("jd_cat_", "category_rules"),
    ("jd_ship_", "shipping"),
    ("jd_srv_", "service_rules"),
    ("jd_rule_", "general_rules"),
    ("jd_ver_", "legacy_archive"),
)
MAX_CHUNK_CHARS = 500
OVERLAP_CHARS = 60
MAX_PARENT_CHARS = 1500


def _infer_doc_group(doc_slug: str) -> str:
    """Infers high-level policy group automatically from document slug prefix."""
    slug_lower = doc_slug.lower()
    for prefix, group in _DOC_GROUP_PREFIXES:
        if slug_lower.startswith(prefix):
            return group
    return "general"


def parse_version_banner(line: str) -> dict[str, Any]:
    """Parses structured temporal/version metadata from a document's visible header banner."""
    stripped = line.strip()
    if not stripped.startswith(_VERSION_LINE_PREFIX):
        return {}
    meta: dict[str, Any] = {"version_banner": stripped}
    ver_match = _VERSION_ID_RE.search(stripped)
    if ver_match:
        meta["version"] = ver_match.group(1).strip()
    from_match = _EFFECTIVE_FROM_RE.search(stripped)
    if from_match:
        meta["effective_from"] = from_match.group(1)
    to_match = _EFFECTIVE_TO_RE.search(stripped)
    if to_match:
        meta["effective_to"] = to_match.group(1)
        meta["effective_to_exclusive"] = True
    is_archived = ("已归档" in stripped) or ("失效" in stripped) or bool(to_match)
    meta["document_status"] = "archived" if is_archived else "current"
    event_match = _APPLICABILITY_EVENT_RE.search(stripped)
    if event_match:
        raw_event = event_match.group(1).strip()
        meta["applicability_event"] = _EVENT_LABEL_TO_CODE.get(raw_event, raw_event)
    return meta


def _normalize_yaml_value(val: Any) -> Any:
    """Converts YAML date/datetime objects recursively into JSON-safe ISO strings."""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    if isinstance(val, dict):
        return {k: _normalize_yaml_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_normalize_yaml_value(v) for v in val]
    return val


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parses and strips leading YAML frontmatter from Markdown text."""
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    try:
        loaded = yaml.safe_load(match.group(1))
        if isinstance(loaded, dict):
            return _normalize_yaml_value(loaded), body
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter, falling back to body only: {e}")
    return {}, body


def _extract_policy_metadata(
    frontmatter: dict[str, Any],
    banner_meta: dict[str, Any],
    doc_slug: str,
) -> dict[str, Any]:
    """Builds normalized chunk metadata automatically from slug, version banner, and optional frontmatter."""
    effective_to = banner_meta.get("effective_to", frontmatter.get("effective_to"))
    default_status = "archived" if effective_to else "current"
    meta: dict[str, Any] = {
        "logical_doc_id": frontmatter.get("id") or frontmatter.get("doc_id") or doc_slug,
        "doc_group": frontmatter.get("category") or _infer_doc_group(doc_slug),
        "version": banner_meta.get("version") or frontmatter.get("version"),
        "document_status": banner_meta.get("document_status") or frontmatter.get("document_status") or default_status,
        "effective_from": banner_meta.get("effective_from") or frontmatter.get("effective_from"),
        "effective_to": effective_to,
        "effective_to_exclusive": banner_meta.get(
            "effective_to_exclusive",
            frontmatter.get("effective_to_exclusive", True),
        ),
        "applicability_event": banner_meta.get("applicability_event") or frontmatter.get("applicability_event"),
        "version_banner": banner_meta.get("version_banner"),
        "timezone": frontmatter.get("timezone", "Asia/Shanghai"),
    }
    # Preserve optional frontmatter filter hints only if an external fixture supplies them
    if frontmatter.get("policy_family"):
        meta["policy_family"] = frontmatter.get("policy_family")
    applicability = frontmatter.get("applicability")
    if isinstance(applicability, dict):
        meta["applicability"] = applicability
        if applicability.get("merchant_types"):
            meta["merchant_types"] = list(applicability["merchant_types"])
    return meta


def document_id_from_path(path: Path, *, root: Path | None = None) -> str:
    """Computes stable document ID from file stem and name hash."""
    stem = path.stem.lower().replace(" ", "_")
    identity_path = path.relative_to(root) if root is not None else Path(path.name)
    digest = hashlib.sha1(identity_path.as_posix().lower().encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    """Formats deterministic chunk ID across ingest runs."""
    return f"{document_id}_{chunk_index:02d}"


def chunk_markdown_file(
    path: Path,
    *,
    root: Path | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    max_parent_chars: int = MAX_PARENT_CHARS,
) -> list[RetrievedChunk]:
    """Reads and chunks a Markdown file into structured RetrievedChunk records."""
    text = path.read_text(encoding="utf-8")
    doc_id = document_id_from_path(path, root=root)
    return chunk_markdown(
        text,
        document_id=doc_id,
        default_title=path.stem,
        doc_slug=path.stem,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
        max_parent_chars=max_parent_chars,
    )


def _split_markdown_blocks(body: str) -> list[tuple[str, str]]:
    """Splits a section body into typed blocks ('table' or 'prose') preserving tables intact."""
    blocks: list[tuple[str, str]] = []
    cur_type: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        nonlocal buf, cur_type
        if buf and cur_type is not None:
            content = "\n".join(buf).strip()
            if content:
                blocks.append((cur_type, content))
        buf = []

    for line in body.splitlines():
        is_table_line = line.lstrip().startswith("|")
        line_type = "table" if is_table_line else "prose"
        if cur_type is None:
            cur_type = line_type
            buf.append(line)
        elif line_type != cur_type:
            # Keep single blank lines inside prose, split when transitioning to/from table
            _flush()
            cur_type = line_type
            buf.append(line)
        else:
            buf.append(line)
    _flush()
    return blocks


def _split_oversized_table(table_text: str, max_size: int) -> list[str]:
    """Splits a Markdown table strictly by row boundaries while repeating header + separator."""
    lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
    if len(table_text) <= max_size or len(lines) <= 3:
        return [table_text]

    header = lines[:2]
    header_prefix = "\n".join(header)
    if len(header_prefix) >= max_size:
        return _sliding_window(table_text, max_size=max_size, overlap=0)

    pieces: list[str] = []
    cur_rows = list(header)
    cur_len = len(header_prefix)
    for row in lines[2:]:
        if cur_len + 1 + len(row) > max_size and len(cur_rows) > 2:
            pieces.append("\n".join(cur_rows))
            cur_rows = list(header) + [row]
            cur_len = len(header_prefix) + 1 + len(row)
        else:
            cur_rows.append(row)
            cur_len += 1 + len(row)
    if len(cur_rows) > 2:
        pieces.append("\n".join(cur_rows))
    return pieces


def _slice_section_preserving_tables(body: str, *, max_size: int, overlap: int) -> list[str]:
    """Chunks section text into <= max_size pieces without corrupting Markdown tables."""
    blocks = _split_markdown_blocks(body)
    if not blocks:
        return []

    pieces: list[str] = []
    pending_prose: list[str] = []

    def _flush_prose() -> None:
        nonlocal pending_prose
        if not pending_prose:
            return
        combined = normalize_text("\n\n".join(pending_prose), join_hard_breaks=True)
        pending_prose = []
        if combined:
            pieces.extend(_sliding_window(combined, max_size=max_size, overlap=overlap))

    for kind, content in blocks:
        if kind == "prose":
            cleaned = normalize_text(content, join_hard_breaks=True)
            if cleaned:
                pending_prose.append(cleaned)
            continue

        # Flush prose before table, or attach short preceding prose if table + prose fit together
        if pending_prose:
            joined_prose = normalize_text("\n\n".join(pending_prose), join_hard_breaks=True)
            if joined_prose and len(joined_prose) + 2 + len(content) <= max_size:
                pending_prose = []
                pieces.append(f"{joined_prose}\n\n{content}")
                continue
            _flush_prose()

        pieces.extend(_split_oversized_table(content, max_size=max_size))

    _flush_prose()
    return [p for p in pieces if p.strip()]


def _build_parent_groups(
    clean_sections: list[tuple[str, str]],
    *,
    doc_slug: str,
    doc_title: str,
    max_parent_chars: int,
    version_banner: str = "",
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Groups consecutive sections into bounded parent blocks for prerequisite/table expansion."""
    groups: list[tuple[str, str, list[tuple[str, str]]]] = []
    cur_items: list[tuple[str, str]] = []
    cur_parts: list[str] = []
    cur_len = 0
    group_idx = 0

    def _flush() -> None:
        nonlocal cur_items, cur_parts, cur_len, group_idx
        if not cur_items:
            return
        pid = f"{doc_slug}::p{group_idx}"
        header_parts = [f"# {doc_title}"] if doc_title else []
        if version_banner:
            header_parts.append(version_banner)
        header = ("\n\n".join(header_parts) + "\n\n") if header_parts else ""
        parent_text = (header + "\n\n".join(cur_parts)).strip()
        groups.append((pid, parent_text, cur_items))
        group_idx += 1
        cur_items = []
        cur_parts = []
        cur_len = 0

    for sec_title, sec_body in clean_sections:
        sec_block = f"## {sec_title}\n{sec_body}"
        if cur_items and cur_len + 2 + len(sec_block) > max_parent_chars:
            _flush()
        cur_items.append((sec_title, sec_body))
        cur_parts.append(sec_block)
        cur_len += (2 if cur_len else 0) + len(sec_block)
    _flush()
    return groups


def chunk_markdown(
    text: str,
    *,
    document_id: str,
    default_title: str = "未命名政策",
    doc_slug: str | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    max_parent_chars: int = MAX_PARENT_CHARS,
) -> list[RetrievedChunk]:
    """Parses headings, header version banners, and optional frontmatter, preserving tables and parent context."""
    frontmatter, body_text = parse_frontmatter(text)
    effective_slug = doc_slug or frontmatter.get("id") or document_id
    fallback_title = str(frontmatter.get("title") or default_title)
    doc_title, sections = _parse_markdown_sections(body_text, default_title=fallback_title)

    # Extract top version banner line & filter boilerplate before parent grouping
    banner_meta: dict[str, Any] = {}
    clean_sections: list[tuple[str, str]] = []
    for section_title, body in sections:
        filtered_lines: list[str] = []
        for ln in body.splitlines():
            stripped = ln.strip()
            if stripped.startswith(_BOILERPLATE_NOTICE_PREFIX):
                continue
            if not banner_meta and stripped.startswith(_VERSION_LINE_PREFIX):
                parsed = parse_version_banner(stripped)
                if parsed:
                    banner_meta = parsed
                    continue
            filtered_lines.append(ln)
        raw_body = "\n".join(filtered_lines).strip()
        if raw_body:
            clean_sections.append((section_title, raw_body))

    policy_meta = _extract_policy_metadata(frontmatter, banner_meta, effective_slug)
    parent_groups = _build_parent_groups(
        clean_sections,
        doc_slug=effective_slug,
        doc_title=doc_title,
        max_parent_chars=max_parent_chars,
        version_banner=str(policy_meta.get("version_banner") or ""),
    )

    chunks: list[RetrievedChunk] = []
    chunk_index = 0
    for parent_id, parent_content, sec_items in parent_groups:
        parent_sections = [s for s, _ in sec_items]
        for section_title, sec_body in sec_items:
            pieces = _slice_section_preserving_tables(
                sec_body,
                max_size=max_chunk_chars,
                overlap=overlap_chars,
            )
            for piece in pieces:
                chunk_meta = {
                    "parser": "markdown_heading_v2",
                    "section": section_title,
                    "doc_slug": effective_slug,
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                    "parent_sections": parent_sections,
                    **policy_meta,
                }
                chunk = RetrievedChunk(
                    document_id=document_id,
                    chunk_id=stable_chunk_id(document_id, chunk_index),
                    title=doc_title,
                    content=piece,
                    source="markdown_ingest",
                    section=section_title,
                    parent_id=parent_id,
                    score=0.0,
                    rank=None,
                    chunk_index=chunk_index,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                chunk_index += 1

    return chunks


def _parse_markdown_sections(text: str, *, default_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Extracts top-level document title (first # Heading) and list of (section_title, body)."""
    lines = text.splitlines()
    doc_title = default_title
    sections: list[tuple[str, str]] = []
    current_section = "概述"
    buffer: list[str] = []

    for line in lines:
        match = _HEADING_PATTERN.match(line.strip())
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            if level == 1 and doc_title == default_title:
                # Top level h1 heading as document title
                doc_title = heading_text
                continue

            # Level 2 or 3 heading starts a new section
            if buffer:
                sections.append((current_section, "\n".join(buffer)))
                buffer = []
            current_section = heading_text
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_section, "\n".join(buffer)))

    return doc_title, (sections or [("正文", text)])


def _sliding_window(text: str, *, max_size: int, overlap: int) -> list[str]:
    """Slices text into chunks with bounded size and overlap."""
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(cleaned) <= max_size:
        return [cleaned] if cleaned else []

    pieces: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_size)
        pieces.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return pieces
