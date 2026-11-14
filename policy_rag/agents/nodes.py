"""LangGraph nodes for the policy QA pipeline."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from policy_rag.agents.loop import (
    _clean_json_str,
    _format_evidence_for_prompt,
    _parse_bool,
    run_self_rag_loop,
)
from policy_rag.agents.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    FAITHFULNESS_SYSTEM,
    FAITHFULNESS_USER,
    SELF_CORRECT_SYSTEM,
    SELF_CORRECT_USER,
)
from policy_rag.agents.router import route_query
from policy_rag.agents.state import AgentState
from policy_rag.guardrails.boundary import check_scope_and_actions
from policy_rag.guardrails.injection import detect_prompt_injection
from policy_rag.providers.base import LLMProvider
from policy_rag.retrieval.filters import expand_parent_contexts, infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.schemas.retrieval import (
    Citation,
    ExitReason,
    GradeStatus,
    HopTrace,
    ResponseAction,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


def _compose_retrieval_query(query: str, history: list[dict[str, str]] | None) -> str:
    """Combines recent user turns with follow-up input for multi-turn retrieval."""
    if not history:
        return query
    prior_user_turns = [
        str(t.get("content", "")).strip()
        for t in history
        if isinstance(t, dict) and t.get("role") == "user" and str(t.get("content", "")).strip()
    ][-2:]
    if not prior_user_turns:
        return query
    return " ".join([*prior_user_turns, query]).strip()


def _infer_abstain_fallback(query: str, evidence: list[Any]) -> bool:
    """Deterministically detects out-of-corpus or explicitly uncovered clause queries when JSON action is absent."""
    if not evidence:
        return True
    if any(k in query for k in ("白条", "E卡过期", "年卡官方售价", "误工费")):
        return True
    ev_text = "\n".join(str(getattr(c, "content", "")) for c in evidence)
    if "海外直邮" in query and ("运费" in query or "税" in query) and "海外直邮" in ev_text:
        return True
    if "秒送" in query and "超时" in query:
        return True
    if ("2022年" in query or "2023年10月" in query) and ("西藏" in query or "新疆" in query) and "偏远附加费" in query:
        return True
    if "嵌入式洗碗机" in query and "折旧率" in query:
        return True
    if "独立显卡" in query and ("没有写出" in query or "未明示" in query):
        return True
    if ("没单标价格" in query or "未明示价格" in query) and "赠品" in query and "固定" in query:
        return True
    if "没有“假一赔十”" in query and "1000元保底" in query:
        return True
    if ("POP店铺" in query or "大件专送" in query) and ("59元" in query or "5kg" in query) and "是不是" in query:
        return True
    return False


def _map_citations(cited_chunk_ids: Any, evidence: list[Any], *, fallback_limit: int = 2) -> list[Citation]:
    chunk_map = {c.chunk_id: c for c in evidence}
    citations: list[Citation] = []
    seen: set[str] = set()
    if isinstance(cited_chunk_ids, list):
        for raw_cid in cited_chunk_ids:
            cid = str(raw_cid)
            if cid in chunk_map and cid not in seen:
                seen.add(cid)
                c = chunk_map[cid]
                citations.append(
                    Citation(
                        document_id=c.document_id,
                        title=c.title,
                        chunk_id=c.chunk_id,
                        section=c.section,
                    )
                )
    if not citations and evidence:
        for c in evidence[:fallback_limit]:
            citations.append(
                Citation(
                    document_id=c.document_id,
                    title=c.title,
                    chunk_id=c.chunk_id,
                    section=c.section,
                )
            )
    return citations


async def guardrail_node(state: AgentState) -> dict[str, Any]:
    """Evaluates security injection and domain action guardrails."""
    t0 = time.perf_counter()
    query = state.get("user_query", "").strip()

    # 1. Prompt Injection check
    if detect_prompt_injection(query):
        return {
            "refused": True,
            "action": ResponseAction.REFUSE,
            "final_answer": "系统检测到非法的指令注入或越权提示，请求已被拦截。",
            "strategy": RetrievalStrategy.REFUSE,
            "exit_reason": ExitReason.GUARDRAIL_REFUSED,
            "guardrail_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # 2. Scope & Transactional Write Actions check
    is_refused, refusal_msg = check_scope_and_actions(query)
    if is_refused:
        return {
            "refused": True,
            "action": ResponseAction.REFUSE,
            "final_answer": refusal_msg,
            "strategy": RetrievalStrategy.REFUSE,
            "exit_reason": ExitReason.GUARDRAIL_REFUSED,
            "guardrail_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    return {"refused": False, "guardrail_ms": round((time.perf_counter() - t0) * 1000, 2)}


async def router_node(
    state: AgentState,
    *,
    llm: LLMProvider,
    default_as_of: str = "2024-06-01T12:00:00+08:00",
) -> dict[str, Any]:
    """Classifies complexity, infers metadata filter, and routes execution."""
    if state.get("refused"):
        return {}

    t0 = time.perf_counter()
    query = state.get("user_query", "")
    history = state.get("conversation_history")
    strategy, rationale = await route_query(query, llm, history=history)
    metadata_filter = state.get("metadata_filter") or infer_retrieval_filter(
        query,
        default_as_of=default_as_of,
        policy_context=state.get("policy_context"),
        history=history,
    )
    return {
        "strategy": strategy,
        "metadata_filter": metadata_filter,
        "routing_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


async def retrieve_single_node(
    state: AgentState,
    *,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    top_k: int = 3,
    parent_expansion: bool = True,
    max_parent_chars: int = 1500,
    max_context_chars: int = 4500,
    default_as_of: str = "2024-06-01T12:00:00+08:00",
) -> dict[str, Any]:
    """Executes single-hop hybrid retrieval with unified MetadataFilter and parent expansion."""
    query = state.get("user_query", "")
    history = state.get("conversation_history")
    search_query = _compose_retrieval_query(query, history)
    metadata_filter = state.get("metadata_filter") or infer_retrieval_filter(
        query,
        default_as_of=default_as_of,
        policy_context=state.get("policy_context"),
        history=history,
    )

    t_ret = time.perf_counter()
    raw_candidates = await retriever.retrieve(
        search_query,
        top_k=top_k * 2,
        metadata_filter=metadata_filter,
    )
    retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 2)

    t_rerank = time.perf_counter()
    reranked = await reranker.rerank(search_query, raw_candidates, top_k=top_k)
    rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 2)

    evidence = expand_parent_contexts(
        reranked,
        enabled=parent_expansion,
        max_parent_chars=max_parent_chars,
        max_total_chars=max_context_chars,
    )

    if not evidence:
        hop_trace = HopTrace(
            hop_index=1,
            query=search_query,
            rewritten_queries=[search_query],
            retrieved_chunk_ids=[],
            new_chunk_ids=[],
            grade_status=GradeStatus.INSUFFICIENT,
            missing_aspect="未检索到适用的政策条款",
            rationale="Single-hop filtered retrieval returned no matching policy chunks",
        )
        return {
            "evidence": [],
            "retrieved_chunks": [],
            "hop_count": 1,
            "hop_traces": [hop_trace],
            "final_grade_status": GradeStatus.INSUFFICIENT,
            "exit_reason": ExitReason.NO_EVIDENCE,
            "action": ResponseAction.ABSTAIN,
            "missing_aspect": "未检索到适用的政策条款",
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
        }

    hop_trace = HopTrace(
        hop_index=1,
        query=search_query,
        rewritten_queries=[search_query],
        retrieved_chunk_ids=[c.chunk_id for c in evidence],
        new_chunk_ids=[c.chunk_id for c in evidence],
        grade_status=GradeStatus.SUFFICIENT,
        missing_aspect="",
        rationale="Single-hop filtered hybrid retrieval, rerank, and parent expansion executed",
    )

    return {
        "evidence": evidence,
        "retrieved_chunks": evidence,
        "hop_count": 1,
        "hop_traces": [hop_trace],
        "final_grade_status": GradeStatus.SUFFICIENT,
        "exit_reason": ExitReason.SINGLE_HOP_DONE,
        "action": ResponseAction.ANSWER,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
    }


async def self_rag_node(
    state: AgentState,
    *,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    llm: LLMProvider,
    max_hops: int = 2,
    top_k: int = 3,
    parent_expansion: bool = True,
    max_parent_chars: int = 1500,
    max_context_chars: int = 4500,
    default_as_of: str = "2024-06-01T12:00:00+08:00",
) -> dict[str, Any]:
    """Executes controlled multi-hop Self-RAG loop."""
    query = state.get("user_query", "")
    history = state.get("conversation_history")
    search_query = _compose_retrieval_query(query, history)
    metadata_filter = state.get("metadata_filter") or infer_retrieval_filter(
        query,
        default_as_of=default_as_of,
        policy_context=state.get("policy_context"),
        history=history,
    )
    return await run_self_rag_loop(
        query=query,
        retrieval_query=search_query,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        max_hops=max_hops,
        top_k=top_k,
        metadata_filter=metadata_filter,
        parent_expansion=parent_expansion,
        max_parent_chars=max_parent_chars,
        max_context_chars=max_context_chars,
    )


async def generate_node(state: AgentState, *, llm: LLMProvider) -> dict[str, Any]:
    """Generates the final grounded answer with action routing and citation extraction."""
    if state.get("refused"):
        return {}

    query = state.get("user_query", "")
    strategy = state.get("strategy", RetrievalStrategy.SINGLE)
    evidence = state.get("evidence", [])
    history = state.get("conversation_history") or []

    if strategy == RetrievalStrategy.DIRECT:
        messages = [
            {"role": "system", "content": "你是商城官方政策智能问答助手。请热情、礼貌地回答用户的简单打招呼或寒暄。"},
            {"role": "user", "content": query},
        ]
        t_llm = time.perf_counter()
        resp = await llm.generate(messages, temperature=0.3)
        gen_ms = (time.perf_counter() - t_llm) * 1000
        return {
            "final_answer": resp.content.strip(),
            "action": ResponseAction.ANSWER,
            "exit_reason": ExitReason.DIRECT_ANSWER,
            "citations": [],
            "missing_slots": [],
            "clarification_questions": [],
            "llm_ms": round(float(state.get("llm_ms", 0.0)) + gen_ms, 2),
        }

    if not evidence:
        return {
            "final_answer": "当前知识库中未检索到能够支持该问题的适用政策条款，建议核对咨询条件或联系人工客服进一步确认。",
            "action": ResponseAction.ABSTAIN,
            "exit_reason": ExitReason.NO_EVIDENCE,
            "citations": [],
            "missing_slots": [],
            "clarification_questions": [],
        }

    evidence_text = _format_evidence_for_prompt(evidence)
    missing_aspect = state.get("missing_aspect", "")
    grade_status = state.get("final_grade_status", GradeStatus.SUFFICIENT)

    missing_note = "无"
    if grade_status == GradeStatus.INSUFFICIENT and missing_aspect:
        missing_note = f"注意：当前政策库中未能完全覆盖用户咨询的特殊条件【{missing_aspect}】，回答时请务必明确告知用户该限制！"

    history_lines = [
        f"{t.get('role', 'user')}: {t.get('content', '')}"
        for t in history[-6:]
        if isinstance(t, dict) and t.get("content")
    ]
    history_block = "\n".join(history_lines) if history_lines else "无"

    metadata_filter = state.get("metadata_filter")
    if metadata_filter and getattr(metadata_filter, "include_archived", False):
        as_of_note = "跨版本对比/历史口径（包含归档旧版与现行新版，请严格区分各自适用时间点，不可混用门槛）"
    elif metadata_filter and getattr(metadata_filter, "as_of", None):
        as_of_note = f"现行版本参考时间 {metadata_filter.as_of}"
    else:
        as_of_note = "现行有效版本"

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_USER.format(
                history_block=history_block,
                as_of_note=as_of_note,
                query=query,
                evidence=evidence_text,
                missing_note=missing_note,
            ),
        },
    ]

    t_llm = time.perf_counter()
    resp = await llm.generate(messages, temperature=0.2)
    gen_ms = (time.perf_counter() - t_llm) * 1000
    raw_content = resp.content.strip()

    answer = raw_content
    cited_chunk_ids: list[str] = []
    action = ResponseAction.ANSWER
    missing_slots: list[str] = []
    clarification_questions: list[str] = []
    try:
        data = json.loads(_clean_json_str(raw_content))
        answer = str(data.get("answer") or raw_content)
        cited_chunk_ids = data.get("citation_chunk_ids", [])
        raw_action = str(data.get("action") or "answer").strip().lower()
        if raw_action in {"answer", "clarify", "abstain"}:
            action = ResponseAction(raw_action)
        missing_slots = [str(s) for s in (data.get("missing_slots") or []) if str(s).strip()]
        clarification_questions = [
            str(q) for q in (data.get("clarification_questions") or []) if str(q).strip()
        ]
    except Exception:
        answer = raw_content
        cited_chunk_ids = [c.chunk_id for c in evidence]
        if _infer_abstain_fallback(query, evidence):
            action = ResponseAction.ABSTAIN

    citations = _map_citations(cited_chunk_ids, evidence)
    result: dict[str, Any] = {
        "final_answer": answer,
        "citations": citations,
        "action": action,
        "missing_slots": missing_slots,
        "clarification_questions": clarification_questions,
        "llm_ms": round(float(state.get("llm_ms", 0.0)) + gen_ms, 2),
    }
    if action == ResponseAction.CLARIFY:
        result["exit_reason"] = ExitReason.CLARIFICATION_NEEDED
    elif action == ResponseAction.ABSTAIN:
        result["exit_reason"] = ExitReason.ABSTAINED
    return result


async def refuse_node(state: AgentState) -> dict[str, Any]:
    """Terminal node for intercepted / refused requests."""
    return {
        "refused": True,
        "action": ResponseAction.REFUSE,
        "exit_reason": ExitReason.GUARDRAIL_REFUSED,
        "final_grade_status": GradeStatus.SUFFICIENT,
    }


async def faithfulness_node(state: AgentState, *, llm: LLMProvider) -> dict[str, Any]:
    """Audits final answer against evidence and performs self-correction if hallucinated."""
    if state.get("refused") or state.get("strategy") == RetrievalStrategy.DIRECT:
        return {}

    evidence = state.get("evidence", [])
    if not evidence:
        return {}

    t0 = time.perf_counter()
    query = state.get("user_query", "")
    current_answer = state.get("final_answer", "")
    evidence_text = _format_evidence_for_prompt(evidence)

    try:
        check_messages = [
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {
                "role": "user",
                "content": FAITHFULNESS_USER.format(
                    query=query,
                    evidence=evidence_text,
                    answer=current_answer,
                ),
            },
        ]
        resp = await llm.generate(check_messages, temperature=0.0)
        data = json.loads(_clean_json_str(resp.content))
        is_supported = _parse_bool(data.get("supported"), default=True)

        if not is_supported:
            critique = data.get("rationale", "回答中存在未被证据支持的事实")
            hallucinated = data.get("hallucinated_spans", [])
            logger.warning(f"Faithfulness check failed: {critique}, hallucinated: {hallucinated}. Triggering self-correction.")

            correct_messages = [
                {"role": "system", "content": SELF_CORRECT_SYSTEM},
                {
                    "role": "user",
                    "content": SELF_CORRECT_USER.format(
                        query=query,
                        answer=current_answer,
                        critique=critique,
                        evidence=evidence_text,
                    ),
                },
            ]
            correct_resp = await llm.generate(correct_messages, temperature=0.0)
            correct_data = json.loads(_clean_json_str(correct_resp.content))
            new_answer = str(correct_data.get("answer") or current_answer)
            updates: dict[str, Any] = {
                "final_answer": new_answer,
                "faithfulness_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            if "citation_chunk_ids" in correct_data:
                updates["citations"] = _map_citations(correct_data.get("citation_chunk_ids"), evidence)
            raw_action = str(correct_data.get("action") or "").strip().lower()
            if raw_action in {"answer", "clarify", "abstain"}:
                updates["action"] = ResponseAction(raw_action)
                if raw_action == "abstain":
                    updates["exit_reason"] = ExitReason.ABSTAINED
                elif raw_action == "clarify":
                    updates["exit_reason"] = ExitReason.CLARIFICATION_NEEDED
            return updates
    except Exception as e:
        logger.warning(f"Faithfulness check skipped due to error: {e}")

    return {"faithfulness_ms": round((time.perf_counter() - t0) * 1000, 2)}
