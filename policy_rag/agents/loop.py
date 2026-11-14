"""Controlled Self-RAG iterative retrieval loop with anti-thrashing mechanisms."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from policy_rag.agents.prompts import (
    GRADE_SYSTEM,
    GRADE_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
)
from policy_rag.providers.base import LLMProvider
from policy_rag.retrieval.filters import expand_parent_contexts, resolve_authority_priority
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.schemas.retrieval import (
    ExitReason,
    GradeStatus,
    HopTrace,
    MetadataFilter,
    ResponseAction,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _clean_json_str(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _format_evidence_for_prompt(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        sec = f" - {c.section}" if c.section else ""
        meta = c.metadata or {}
        ver_banner = str(meta.get("version_banner") or "").strip()
        ver_tag = f" | {ver_banner}" if ver_banner and ver_banner not in c.content else ""
        _, authority_label = resolve_authority_priority(c)
        if c.source == "clause_graph" or meta.get("graph_expanded"):
            relation = str(meta.get("graph_edge") or "1-Hop关联")
            recall_source = f"图谱1-Hop关联补全 ({relation})"
        elif c.source:
            recall_source = c.source
        else:
            recall_source = "直接召回"
        lines.append(
            f"[{c.chunk_id}] 《{c.title}》{sec}{ver_tag} | "
            f"[效力优先级: {authority_label}] [召回来源: {recall_source}]:\n{c.content}\n"
        )
    return "\n".join(lines)


async def run_self_rag_loop(
    query: str,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    llm: LLMProvider,
    max_hops: int = 2,
    top_k: int = 3,
    *,
    retrieval_query: str | None = None,
    metadata_filter: MetadataFilter | None = None,
    parent_expansion: bool = True,
    max_parent_chars: int = 1500,
    max_context_chars: int = 4500,
) -> dict[str, Any]:
    """Executes a controlled multi-hop retrieval loop.

    Key safeguards:
    1. 3-state grading (SUFFICIENT / INSUFFICIENT / GRADER_UNAVAILABLE).
    2. Anti-thrashing progress check: if no new unique chunks retrieved in a hop -> NO_NEW_EVIDENCE exit.
    3. Max hops guardrail -> MAX_HOPS_REACHED exit.
    4. Graceful degradation on LLM JSON/network errors -> GRADER_UNAVAILABLE with accumulated evidence.
    """
    accumulated_chunks: dict[str, RetrievedChunk] = {}
    hop_traces: list[HopTrace] = []
    all_rewritten_queries: list[str] = []
    final_grade_status = GradeStatus.INSUFFICIENT
    exit_reason = ExitReason.MAX_HOPS_REACHED
    missing_aspect = ""
    retrieval_ms = 0.0
    rerank_ms = 0.0
    llm_ms = 0.0

    current_query = (retrieval_query or query).strip()

    for hop in range(1, max_hops + 1):
        # 1. Determine queries for this hop
        hop_queries: list[str] = []
        if hop == 1:
            hop_queries = [current_query]
        else:
            # Query rewrite based on missing_aspect
            try:
                rewrite_prompt = [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": REWRITE_USER.format(query=query, missing_aspect=missing_aspect)},
                ]
                t_llm = time.perf_counter()
                resp = await llm.generate(rewrite_prompt, temperature=0.1)
                llm_ms += (time.perf_counter() - t_llm) * 1000
                data = json.loads(_clean_json_str(resp.content))
                hop_queries = [q.strip() for q in data.get("queries", []) if q.strip()]
            except Exception as e:
                logger.warning(f"Query rewrite failed at hop {hop}: {e}")
                hop_queries = [f"{query} {missing_aspect}".strip()]

            if not hop_queries:
                hop_queries = [f"{query} {missing_aspect}".strip()]

            all_rewritten_queries.extend(hop_queries)

        # 2. Retrieve & Rerank for each query in this hop with shared pre-fusion filter
        hop_retrieved_map: dict[str, RetrievedChunk] = {}
        for q in hop_queries:
            t_ret = time.perf_counter()
            raw_candidates = await retriever.retrieve(
                q,
                top_k=top_k * 2,
                metadata_filter=metadata_filter,
            )
            retrieval_ms += (time.perf_counter() - t_ret) * 1000

            t_rerank = time.perf_counter()
            reranked = await reranker.rerank(q, raw_candidates, top_k=top_k)
            rerank_ms += (time.perf_counter() - t_rerank) * 1000
            for c in reranked:
                if c.chunk_id not in hop_retrieved_map:
                    hop_retrieved_map[c.chunk_id] = c

        retrieved_ids = list(hop_retrieved_map.keys())
        new_chunk_ids = [cid for cid in retrieved_ids if cid not in accumulated_chunks]

        # 3. Anti-thrashing progress check
        if hop > 1 and len(new_chunk_ids) == 0:
            exit_reason = ExitReason.NO_NEW_EVIDENCE
            hop_traces.append(
                HopTrace(
                    hop_index=hop,
                    query=current_query,
                    rewritten_queries=hop_queries,
                    retrieved_chunk_ids=retrieved_ids,
                    new_chunk_ids=[],
                    grade_status=final_grade_status,
                    missing_aspect=missing_aspect,
                    rationale="本轮未检索到任何去重增量证据，提前终止补检以防止死循环空耗。",
                )
            )
            break

        # Accumulate new chunks
        for cid in new_chunk_ids:
            accumulated_chunks[cid] = hop_retrieved_map[cid]

        # 4. Grade accumulated evidence (after parent expansion so intact tables/clauses are visible)
        raw_evidence_list = list(accumulated_chunks.values())
        if not raw_evidence_list:
            missing_aspect = "未检索到适用的政策条款"
            hop_traces.append(
                HopTrace(
                    hop_index=hop,
                    query=current_query,
                    rewritten_queries=hop_queries if hop > 1 else [],
                    retrieved_chunk_ids=[],
                    new_chunk_ids=[],
                    grade_status=GradeStatus.INSUFFICIENT,
                    missing_aspect=missing_aspect,
                    rationale="本轮未检索到任何匹配筛选条件的证据片段。",
                )
            )
            continue

        graded_evidence_list = expand_parent_contexts(
            raw_evidence_list,
            enabled=parent_expansion,
            max_parent_chars=max_parent_chars,
            max_total_chars=max_context_chars,
        )
        evidence_text = _format_evidence_for_prompt(graded_evidence_list)

        try:
            grade_prompt = [
                {"role": "system", "content": GRADE_SYSTEM},
                {"role": "user", "content": GRADE_USER.format(query=query, evidence=evidence_text)},
            ]
            t_grade = time.perf_counter()
            resp = await llm.generate(grade_prompt, temperature=0.0)
            llm_ms += (time.perf_counter() - t_grade) * 1000
            grade_data = json.loads(_clean_json_str(resp.content))
            is_sufficient = _parse_bool(grade_data.get("sufficient"), default=False)
            missing_aspect = str(grade_data.get("missing_aspect", "")).strip()
            rationale = str(grade_data.get("rationale", ""))

            if is_sufficient:
                final_grade_status = GradeStatus.SUFFICIENT
                exit_reason = ExitReason.SUFFICIENT_EVIDENCE
                hop_traces.append(
                    HopTrace(
                        hop_index=hop,
                        query=current_query,
                        rewritten_queries=hop_queries if hop > 1 else [],
                        retrieved_chunk_ids=retrieved_ids,
                        new_chunk_ids=new_chunk_ids,
                        grade_status=GradeStatus.SUFFICIENT,
                        missing_aspect="",
                        rationale=rationale,
                    )
                )
                break
            else:
                final_grade_status = GradeStatus.INSUFFICIENT
                hop_traces.append(
                    HopTrace(
                        hop_index=hop,
                        query=current_query,
                        rewritten_queries=hop_queries if hop > 1 else [],
                        retrieved_chunk_ids=retrieved_ids,
                        new_chunk_ids=new_chunk_ids,
                        grade_status=GradeStatus.INSUFFICIENT,
                        missing_aspect=missing_aspect,
                        rationale=rationale,
                    )
                )
        except Exception as e:
            logger.warning(f"Evidence grading failed or returned invalid JSON at hop {hop}: {e}")
            final_grade_status = GradeStatus.GRADER_UNAVAILABLE
            exit_reason = ExitReason.SUFFICIENT_EVIDENCE  # Degrade gracefully to answer with available evidence
            hop_traces.append(
                HopTrace(
                    hop_index=hop,
                    query=current_query,
                    rewritten_queries=hop_queries if hop > 1 else [],
                    retrieved_chunk_ids=retrieved_ids,
                    new_chunk_ids=new_chunk_ids,
                    grade_status=GradeStatus.GRADER_UNAVAILABLE,
                    missing_aspect=missing_aspect,
                    rationale=f"评审模型调用异常或JSON损坏，降级终止补检: {e}",
                )
            )
            break

    final_evidence = expand_parent_contexts(
        list(accumulated_chunks.values()),
        enabled=parent_expansion,
        max_parent_chars=max_parent_chars,
        max_total_chars=max_context_chars,
    )

    if not final_evidence:
        return {
            "evidence": [],
            "hop_traces": hop_traces,
            "hop_count": len(hop_traces),
            "final_grade_status": GradeStatus.INSUFFICIENT,
            "exit_reason": ExitReason.NO_EVIDENCE,
            "action": ResponseAction.ABSTAIN,
            "missing_aspect": missing_aspect or "未检索到适用的政策条款",
            "rewritten_queries": all_rewritten_queries,
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "llm_ms": round(llm_ms, 2),
        }

    # If completed all hops without early exit
    if exit_reason not in {ExitReason.SUFFICIENT_EVIDENCE, ExitReason.NO_NEW_EVIDENCE}:
        exit_reason = ExitReason.MAX_HOPS_REACHED

    return {
        "evidence": final_evidence,
        "hop_traces": hop_traces,
        "hop_count": len(hop_traces),
        "final_grade_status": final_grade_status,
        "exit_reason": exit_reason,
        "action": ResponseAction.ANSWER,
        "missing_aspect": missing_aspect,
        "rewritten_queries": all_rewritten_queries,
        "retrieval_ms": round(retrieval_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "llm_ms": round(llm_ms, 2),
    }
