"""Shared metadata filtering and parent-context expansion utilities."""

from __future__ import annotations

import re
from typing import Any

from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk

_HISTORICAL_YEAR_RE = re.compile(r"(?:20(?:1[6-9]|2[0-3]))")
_CURRENT_YEAR_RE = re.compile(r"(?:202[4-9])")
_CORRECTION_CUES = ("不对", "看错", "更正", "写错", "搞错", "改一下", "修正为")
_VERSION_CONTRAST_CUES = (
    "旧版",
    "旧单",
    "旧规",
    "旧流程",
    "旧门槛",
    "旧规则",
    "老流程",
    "老版本",
    "老门槛",
    "老标准",
    "老规则",
    "以前",
    "生效前",
    "新旧",
    "新版",
    "追溯",
    "满99",
    "99元门槛",
    "5张券",
    "每月5张",
    "先垫",
    "寄回复核",
    "12月31日",
    "以前用完",
)


def _extract_date_prefix(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    return None


def matches_metadata_filter(
    chunk: RetrievedChunk,
    metadata_filter: MetadataFilter | None,
) -> bool:
    """Evaluate whether a chunk satisfies shared pre-fusion metadata constraints."""
    if metadata_filter is None:
        return True

    meta = chunk.metadata or {}
    doc_status = str(meta.get("document_status", "current")).lower()

    if not metadata_filter.include_archived:
        if doc_status == "archived":
            return False

        as_of_date = _extract_date_prefix(metadata_filter.as_of)
        if as_of_date:
            eff_from = _extract_date_prefix(meta.get("effective_from"))
            eff_to = _extract_date_prefix(meta.get("effective_to"))
            exclusive = bool(meta.get("effective_to_exclusive", True))

            if eff_from and as_of_date < eff_from:
                return False
            if eff_to:
                if exclusive and as_of_date >= eff_to:
                    return False
                if not exclusive and as_of_date > eff_to:
                    return False

    if metadata_filter.doc_groups:
        doc_group = str(meta.get("doc_group") or "")
        if doc_group and doc_group not in metadata_filter.doc_groups:
            return False

    if metadata_filter.policy_families:
        family = str(meta.get("policy_family") or "")
        if family and family not in metadata_filter.policy_families:
            return False

    if metadata_filter.merchant_types:
        allowed_merchants = meta.get("merchant_types")
        if isinstance(allowed_merchants, list) and allowed_merchants:
            if not any(m in allowed_merchants for m in metadata_filter.merchant_types):
                return False

    return True


def infer_retrieval_filter(
    query: str,
    *,
    default_as_of: str = "2026-01-01",
    policy_context: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> MetadataFilter:
    """Infer temporal, category, and version-scope filter from query, history, and policy_context.

    By default, archived policy documents are excluded so active queries cannot mix
    deprecated thresholds with current rules. When explicit historical dates or
    version-contrast cues appear, archived documents are unlocked alongside current rules.
    """
    # If user explicitly corrects a previous historical year to a current-era year without version comparison,
    # prevent stale historical turns in history from reopening archived documents.
    is_temporal_correction = (
        any(c in query for c in _CORRECTION_CUES)
        and bool(_CURRENT_YEAR_RE.search(query))
        and not _HISTORICAL_YEAR_RE.search(query)
        and not any(cue in query for cue in _VERSION_CONTRAST_CUES)
    )
    parts = [query]
    if history and not is_temporal_correction:
        for turn in history[-4:]:
            content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
            if content:
                parts.append(content)
    combined = " ".join(parts)

    doc_groups = None
    policy_families = None
    merchant_types = None
    if isinstance(policy_context, dict):
        if policy_context.get("doc_groups"):
            doc_groups = [str(g) for g in policy_context["doc_groups"]]
        if policy_context.get("policy_families"):
            policy_families = [str(f) for f in policy_context["policy_families"]]
        if policy_context.get("merchant_types"):
            merchant_types = [str(m) for m in policy_context["merchant_types"]]

    wants_historical_or_contrast = bool(_HISTORICAL_YEAR_RE.search(combined)) or any(
        cue in combined for cue in _VERSION_CONTRAST_CUES
    )
    if wants_historical_or_contrast:
        return MetadataFilter(
            as_of=None,
            include_archived=True,
            doc_groups=doc_groups,
            policy_families=policy_families,
            merchant_types=merchant_types,
        )

    ctx_as_of = None
    if isinstance(policy_context, dict) and policy_context.get("as_of"):
        ctx_as_of = str(policy_context["as_of"])
    return MetadataFilter(
        as_of=ctx_as_of or default_as_of,
        include_archived=False,
        doc_groups=doc_groups,
        policy_families=policy_families,
        merchant_types=merchant_types,
    )


def resolve_authority_priority(chunk: RetrievedChunk) -> tuple[int, str]:
    """Resolves Lex Specialis priority tier and display label for a policy chunk."""
    meta = chunk.metadata or {}
    doc_group = str(meta.get("doc_group") or "").lower()
    doc_status = str(meta.get("document_status") or "").lower()
    logical_id = str(meta.get("logical_doc_id") or chunk.document_id or "").lower()

    if (
        doc_status == "archived"
        or doc_group in {"service_rules", "legacy_archive", "legacy_rules"}
        or logical_id.startswith(("jd_srv_", "jd_ver_"))
    ):
        return 1, "P1-专项服务与特定版本规则（最高优先适用）"
    if doc_group in {"category_rules", "shipping"} or logical_id.startswith(("jd_cat_", "jd_ship_")):
        return 2, "P2-品类特例与专项运费细则（优先于通用总则）"
    return 3, "P3-平台通用基础总则（无特例冲突时适用）"


def _lost_in_the_middle_reorder(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Places highest-priority chunks at start and end of prompt, lower-priority in the middle."""
    if len(chunks) <= 2:
        return list(chunks)
    left: list[RetrievedChunk] = []
    right: list[RetrievedChunk] = []
    for idx, chunk in enumerate(chunks):
        if idx % 2 == 0:
            left.append(chunk)
        else:
            right.append(chunk)
    right.reverse()
    return left + right


def expand_parent_contexts(
    chunks: list[RetrievedChunk],
    *,
    enabled: bool = True,
    max_parent_chars: int = 1500,
    max_total_chars: int = 6000,
    reorder_lost_in_the_middle: bool = True,
) -> list[RetrievedChunk]:
    """Deduplicate retrieved child hits by parent_id, annotate Lex Specialis priority, and attach parent context."""
    if not chunks:
        return []
    if not enabled:
        copies = []
        for c in chunks:
            copy_chunk = c.model_copy(deep=True)
            tier, label = resolve_authority_priority(copy_chunk)
            copy_chunk.metadata = dict(copy_chunk.metadata or {})
            copy_chunk.metadata["authority_tier"] = tier
            copy_chunk.metadata["authority_label"] = label
            copies.append(copy_chunk)
        return copies

    expanded: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    used_chars = 0

    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        parent_key = chunk.parent_id or str(meta.get("parent_id") or "") or chunk.chunk_id
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)

        copy_chunk = chunk.model_copy(deep=True)
        copy_chunk.metadata = dict(copy_chunk.metadata or {})
        parent_text = str(meta.get("parent_content") or "").strip()
        candidate_text = parent_text[:max_parent_chars] if parent_text else copy_chunk.content

        if expanded and used_chars + len(candidate_text) > max_total_chars:
            # Preserve diversity of top hits within context budget by falling back to child chunk
            candidate_text = copy_chunk.content[:max_parent_chars]
            if used_chars + len(candidate_text) > max_total_chars:
                break

        if parent_text and candidate_text != copy_chunk.content:
            copy_chunk.metadata["child_content"] = copy_chunk.content
            copy_chunk.content = candidate_text
            copy_chunk.metadata["expanded_from_parent"] = True

        tier, label = resolve_authority_priority(copy_chunk)
        copy_chunk.metadata["authority_tier"] = tier
        copy_chunk.metadata["authority_label"] = label

        expanded.append(copy_chunk)
        used_chars += len(copy_chunk.content)

    if reorder_lost_in_the_middle and len(expanded) > 1:
        prioritized = [
            c
            for _, c in sorted(
                enumerate(expanded),
                key=lambda item: (int((item[1].metadata or {}).get("authority_tier", 3)), item[0]),
            )
        ]
        return _lost_in_the_middle_reorder(prioritized)

    return expanded
