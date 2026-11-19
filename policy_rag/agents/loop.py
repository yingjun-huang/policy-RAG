"""Controlled Self-RAG iterative retrieval loop with anti-thrashing mechanisms."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from policy_rag.agents.contracts import EvidenceGrade, RewritePayload, parse_model_bool as _parse_bool
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
    query: str, retriever: HybridRetriever, reranker: DocumentReranker, llm: LLMProvider,
    max_hops: int = 2, top_k: int = 3, *, retrieval_query: str | None = None,
    metadata_filter: MetadataFilter | None = None, parent_expansion: bool = True,
    max_parent_chars: int = 1500, max_context_chars: int = 4500,
) -> dict[str, Any]:
    """One or two retrieval rounds, with explicit policy vs user-fact gaps."""
    accumulated: dict[str, RetrievedChunk] = {}
    traces: list[HopTrace] = []
    rewrites: list[str] = []
    grade = GradeStatus.INSUFFICIENT
    exit_reason = ExitReason.MAX_HOPS_REACHED
    missing_aspect = ""
    missing_type = "policy_evidence"
    missing_slots: list[str] = []
    questions: list[str] = []
    action = ResponseAction.ANSWER
    retrieval_ms = rerank_ms = llm_ms = 0.0
    error = None
    first_query = (retrieval_query or query).strip()
    for hop in range(1, max_hops + 1):
        hop_queries = [first_query]
        if hop > 1:
            started = time.perf_counter()
            try:
                response = await llm.generate([
                    {"role":"system","content":REWRITE_SYSTEM},
                    {"role":"user","content":REWRITE_USER.format(query=query,missing_aspect=missing_aspect)},
                ], temperature=0.1, max_tokens=512, response_format={"type":"json_object"})
                payload = RewritePayload.model_validate(json.loads(_clean_json_str(response.content)))
                hop_queries = list(dict.fromkeys(payload.queries))
            except Exception as exc:
                logger.warning("Query rewrite unavailable: %s",exc)
                hop_queries = [f"{query} {missing_aspect}".strip()]
            finally:
                llm_ms += (time.perf_counter()-started)*1000
            rewrites.extend(hop_queries)

        retrieved: dict[str, RetrievedChunk] = {}
        for search_query in hop_queries:
            started=time.perf_counter()
            raw=await retriever.retrieve(search_query,top_k=top_k*2,metadata_filter=metadata_filter)
            retrieval_ms+=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            ranked=await reranker.rerank(search_query,raw,top_k=top_k)
            rerank_ms+=(time.perf_counter()-started)*1000
            for c in ranked:retrieved.setdefault(c.chunk_id,c)
        ids=list(retrieved)
        new_ids=[cid for cid in ids if cid not in accumulated]
        if hop>1 and not new_ids:
            exit_reason=ExitReason.NO_NEW_EVIDENCE
            traces.append(HopTrace(hop_index=hop,query=first_query,rewritten_queries=hop_queries,
                retrieved_chunk_ids=ids,new_chunk_ids=[],grade_status=grade,missing_aspect=missing_aspect,
                rationale="本轮未检索到去重增量证据，停止补检。"))
            break
        for cid in new_ids:accumulated[cid]=retrieved[cid]
        if not accumulated:
            missing_aspect="未检索到适用的政策条款"
            traces.append(HopTrace(hop_index=hop,query=first_query,rewritten_queries=hop_queries,
                retrieved_chunk_ids=[],new_chunk_ids=[],grade_status=GradeStatus.INSUFFICIENT,
                missing_aspect=missing_aspect,rationale="本轮没有匹配筛选条件的证据。"))
            continue

        evidence=expand_parent_contexts(list(accumulated.values()),enabled=parent_expansion,
            max_parent_chars=max_parent_chars,max_total_chars=max_context_chars)
        started=time.perf_counter()
        try:
            response=await llm.generate([
                {"role":"system","content":GRADE_SYSTEM},
                {"role":"user","content":GRADE_USER.format(query=query,evidence=_format_evidence_for_prompt(evidence))},
            ],temperature=0.0,max_tokens=1024,response_format={"type":"json_object"})
            decision=EvidenceGrade.model_validate(json.loads(_clean_json_str(response.content)))
            missing_type=decision.missing_type
            missing_aspect=decision.missing_aspect
            missing_slots=decision.missing_slots
            questions=decision.clarification_questions
            grade=GradeStatus.SUFFICIENT if decision.sufficient else GradeStatus.INSUFFICIENT
            rationale=decision.rationale
            if decision.sufficient:
                exit_reason=ExitReason.SUFFICIENT_EVIDENCE
            elif missing_type in {"user_facts","both"}:
                action=ResponseAction.CLARIFY
                exit_reason=ExitReason.CLARIFICATION_NEEDED
        except Exception as exc:
            logger.warning("Evidence grading unavailable: %s",exc)
            grade=GradeStatus.GRADER_UNAVAILABLE
            exit_reason=ExitReason.GRADER_UNAVAILABLE
            action=ResponseAction.ABSTAIN
            error="grader_unavailable"
            rationale="证据评审不可用，停止补检并保守留白。"
        finally:
            llm_ms+=(time.perf_counter()-started)*1000
        traces.append(HopTrace(hop_index=hop,query=first_query,
            rewritten_queries=hop_queries if hop>1 else [],retrieved_chunk_ids=ids,new_chunk_ids=new_ids,
            grade_status=grade,missing_aspect=missing_aspect,rationale=rationale))
        if grade in {GradeStatus.SUFFICIENT,GradeStatus.GRADER_UNAVAILABLE} or action==ResponseAction.CLARIFY:
            break

    evidence=expand_parent_contexts(list(accumulated.values()),enabled=parent_expansion,
        max_parent_chars=max_parent_chars,max_total_chars=max_context_chars)
    if not evidence:
        grade=GradeStatus.INSUFFICIENT
        exit_reason=ExitReason.NO_EVIDENCE
        action=ResponseAction.ABSTAIN
        missing_aspect=missing_aspect or "未检索到适用的政策条款"
    return {"evidence":evidence,"retrieved_chunks":evidence,"hop_traces":traces,"hop_count":len(traces),
        "final_grade_status":grade,"exit_reason":exit_reason,"action":action,
        "missing_type":missing_type,"missing_aspect":missing_aspect,"missing_slots":missing_slots,
        "clarification_questions":questions,"rewritten_queries":rewrites,"error":error,
        "retrieval_ms":round(retrieval_ms,2),"rerank_ms":round(rerank_ms,2),"llm_ms":round(llm_ms,2)}
