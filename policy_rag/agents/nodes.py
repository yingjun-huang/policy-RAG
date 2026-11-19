"""LangGraph nodes for the policy QA pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal

from policy_rag.agents.contracts import AnswerPayload as _AnswerPayload

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
from policy_rag.calculation import CalculationRequest, resolve_calculations
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
    ][-3:]
    if not prior_user_turns:
        return query
    return " ".join([*prior_user_turns, query]).strip()


def _map_citations(cited_chunk_ids: Any, evidence: list[Any]) -> list[Citation]:
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
    state: AgentState, *, retriever: HybridRetriever, reranker: DocumentReranker, llm: LLMProvider,
    top_k: int = 3, parent_expansion: bool = True, max_parent_chars: int = 1500,
    max_context_chars: int = 4500, default_as_of: str = "2024-06-01T12:00:00+08:00",
) -> dict[str, Any]:
    """Reuse evidence grading with a one-round retrieval budget."""
    query=state.get("user_query","")
    history=state.get("conversation_history")
    search_query=_compose_retrieval_query(query,history)
    metadata_filter=state.get("metadata_filter") or infer_retrieval_filter(query,
        default_as_of=default_as_of,policy_context=state.get("policy_context"),history=history)
    result=await run_self_rag_loop(query=search_query,retrieval_query=search_query,
        retriever=retriever,reranker=reranker,llm=llm,max_hops=1,top_k=top_k,
        metadata_filter=metadata_filter,parent_expansion=parent_expansion,
        max_parent_chars=max_parent_chars,max_context_chars=max_context_chars)
    if result["exit_reason"]==ExitReason.SUFFICIENT_EVIDENCE:
        result["exit_reason"]=ExitReason.SINGLE_HOP_DONE
    return result


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
        query=search_query,
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

    if state.get("action") == ResponseAction.CLARIFY and state.get("missing_type") in {"user_facts","both"}:
        questions=state.get("clarification_questions",[])
        return {"final_answer":"为判断适用条款，请补充："+"；".join(questions),
                "action":ResponseAction.CLARIFY,"exit_reason":ExitReason.CLARIFICATION_NEEDED,
                "missing_slots":state.get("missing_slots",[]),"clarification_questions":questions,
                "citations":[],"calculation_trace":[]}
    if state.get("error") == "grader_unavailable":
        return {"final_answer":"证据评审暂不可用，暂时无法给出可靠结论，请稍后重试。",
                "action":ResponseAction.ABSTAIN,"exit_reason":ExitReason.GRADER_UNAVAILABLE,
                "citations":[],"calculation_trace":[]}

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
    if metadata_filter and getattr(metadata_filter, "as_of", None):
        as_of_note = f"适用版本参考时间 {metadata_filter.as_of}，仅采用该日期生效的条款"
    elif metadata_filter and getattr(metadata_filter, "include_archived", False):
        as_of_note = "跨版本对比/历史口径（包含归档旧版与现行新版，请严格区分各自适用时间点，不可混用门槛）"
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
    gen_ms = 0.0
    try:
        resp = await llm.generate(messages, temperature=0.2, response_format={"type":"json_object"})
        gen_ms = (time.perf_counter() - t_llm) * 1000
        raw_content = resp.content.strip()
        data = _AnswerPayload.model_validate(json.loads(_clean_json_str(raw_content)))
        answer, calculation_trace = resolve_calculations(data.answer.strip(), data.calculations, evidence)
        if not answer:
            raise ValueError("Empty model answer")
        action = ResponseAction(data.action)
        cited_chunk_ids = list(dict.fromkeys(data.citation_chunk_ids + [cid for calc in data.calculations for cid in calc.evidence_chunk_ids]))
        missing_slots = data.missing_slots if action == ResponseAction.CLARIFY else []
        clarification_questions = data.clarification_questions if action == ResponseAction.CLARIFY else []
    except Exception as exc:
        gen_ms = (time.perf_counter() - t_llm) * 1000
        logger.warning("Answer response unavailable: %s", exc)
        return {
            "final_answer": "本次回答生成异常，暂时无法给出可靠结论，请稍后重试。",
            "action": ResponseAction.ABSTAIN,
            "exit_reason": ExitReason.ABSTAINED,
            "citations": [], "missing_slots": [], "clarification_questions": [],
            "error": "generation_invalid_response",
            "llm_ms": round(float(state.get("llm_ms", 0.0)) + gen_ms, 2),
        }

    citations = _map_citations(cited_chunk_ids, evidence)
    result: dict[str, Any] = {
        "final_answer": answer,
        "calculation_trace": calculation_trace,
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
        "final_answer": state.get("final_answer") or "本系统仅提供商城政策咨询，无法查询或办理具体订单业务。请前往商城 App 的【我的订单】或联系人工客服。",
        "exit_reason": ExitReason.GUARDRAIL_REFUSED,
        "final_grade_status": GradeStatus.SUFFICIENT,
    }


_HIGH_RISK_FACT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:元|%|％|折|秒|分钟|小时|个?工作日|个?自然日|天|日|周|个?月|年|kg|千克|克|cm|厘米|倍|京豆|张)"
    r"|[零〇一二两三四五六七八九十百千万]+(?:秒|分钟|小时|个?工作日|个?自然日|天|日|周|个?月|年)"
    r"|(?:20\d{2}年|\d{4}-\d{2}-\d{2})"
)


def should_run_faithfulness(state: AgentState) -> tuple[bool, str]:
    """Determines whether Faithfulness LLM audit is warranted under risk-gated policy."""
    if state.get("refused") or state.get("strategy") == RetrievalStrategy.DIRECT:
        return False, "skipped:non_retrieval"

    if state.get("error"):
        return False, "skipped:generation_error"

    if not state.get("evidence"):
        return False, "skipped:no_evidence"

    action = state.get("action")
    if action == ResponseAction.CLARIFY:
        return False, "skipped:clarify_action"

    grade_status = state.get("final_grade_status", GradeStatus.SUFFICIENT)
    if grade_status != GradeStatus.SUFFICIENT:
        return True, "risk:insufficient_or_unavailable_grade"

    if action == ResponseAction.ABSTAIN:
        return True, "risk:abstain_verification"

    answer_text = str(state.get("final_answer") or "")
    if _HIGH_RISK_FACT_RE.search(answer_text):
        return True, "risk:numeric_or_temporal_claim"

    return False, "skipped:low_risk_qualitative"


async def faithfulness_node(
    state: AgentState,
    *,
    llm: LLMProvider,
    risk_gate_enabled: bool = True,
) -> dict[str, Any]:
    """Audits high-risk final answers against evidence and performs self-correction if hallucinated."""
    if state.get("refused") or state.get("strategy") == RetrievalStrategy.DIRECT:
        return {
            "faithfulness_checked": False,
            "faithfulness_gate_reason": "skipped:non_retrieval",
        }

    if state.get("error"):
        return {"faithfulness_checked":False,"faithfulness_triggered":False,
                "faithfulness_gate_reason":"skipped:generation_error"}
    if state.get("missing_type") in {"user_facts","both"} and state.get("action")==ResponseAction.CLARIFY:
        return {"faithfulness_checked":False,"faithfulness_triggered":False,
                "faithfulness_gate_reason":"skipped:pending_user_facts"}

    evidence = state.get("evidence", [])
    if not evidence:
        return {
            "faithfulness_checked": False,
            "faithfulness_gate_reason": "skipped:no_evidence",
        }

    if risk_gate_enabled:
        should_check, gate_reason = should_run_faithfulness(state)
        if not should_check:
            return {
                "faithfulness_checked": False,
                "faithfulness_gate_reason": gate_reason,
                "faithfulness_triggered": False,
                "faithfulness_ms": 0.0,
            }
    else:
        gate_reason = "full_audit_enabled"

    t0 = time.perf_counter()
    query = _compose_retrieval_query(state.get("user_query", ""), state.get("conversation_history"))
    current_answer = state.get("final_answer", "")
    evidence_text = _format_evidence_for_prompt(evidence)

    is_supported = None
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
        is_supported = _parse_bool(data.get("supported"))

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
            payload = _AnswerPayload.model_validate(correct_data)
            corrected_answer, calculation_trace = resolve_calculations(payload.answer.strip(), payload.calculations, evidence)
            if not corrected_answer:
                raise ValueError("Empty correction answer")
            action = ResponseAction(payload.action)
            updates: dict[str, Any] = {
                "final_answer": corrected_answer,
                "calculation_trace": calculation_trace,
                "action": action,
                "citations": _map_citations(list(dict.fromkeys(payload.citation_chunk_ids + [cid for calc in payload.calculations for cid in calc.evidence_chunk_ids])), evidence),
                "missing_slots": payload.missing_slots if action == ResponseAction.CLARIFY else [],
                "clarification_questions": payload.clarification_questions if action == ResponseAction.CLARIFY else [],
                "faithfulness_checked": True,
                "faithfulness_gate_reason": gate_reason,
                "faithfulness_triggered": True,
                "pre_correction_answer": current_answer,
                "pre_correction_action": state.get("action", ResponseAction.ANSWER),
                "faithfulness_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            if action == ResponseAction.ABSTAIN:
                updates["exit_reason"] = ExitReason.ABSTAINED
            elif action == ResponseAction.CLARIFY:
                updates["exit_reason"] = ExitReason.CLARIFICATION_NEEDED
            elif state.get("exit_reason") in {ExitReason.ABSTAINED, ExitReason.CLARIFICATION_NEEDED}:
                if state.get("strategy") != RetrievalStrategy.MULTI:
                    updates["exit_reason"] = ExitReason.SINGLE_HOP_DONE
                else:
                    updates["exit_reason"] = (
                        ExitReason.SUFFICIENT_EVIDENCE
                        if state.get("final_grade_status") == GradeStatus.SUFFICIENT
                        else ExitReason.MAX_HOPS_REACHED
                    )
            return updates
    except Exception as e:
        logger.warning(f"Faithfulness check/correction unavailable: {e}")
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        if is_supported is False:
            return {
                "final_answer": "当前回答未通过证据核验，且修正未完成，暂时无法给出可靠结论，请稍后重试。",
                "calculation_trace": [],
                "action": ResponseAction.ABSTAIN, "exit_reason": ExitReason.ABSTAINED,
                "citations": [], "missing_slots": [], "clarification_questions": [],
                "pre_correction_answer": current_answer,
                "pre_correction_action": state.get("action", ResponseAction.ANSWER),
                "faithfulness_checked": True, "faithfulness_triggered": True,
                "faithfulness_gate_reason": f"{gate_reason}:correction_failed",
                "faithfulness_ms": elapsed, "error": "self_correction_failed",
            }
        return {
            "faithfulness_checked": False, "faithfulness_triggered": False,
            "faithfulness_gate_reason": "unavailable:checker_error",
            "faithfulness_ms": elapsed, "error": "faithfulness_unavailable",
        }

    return {
        "faithfulness_checked": True,
        "faithfulness_gate_reason": gate_reason,
        "faithfulness_triggered": False,
        "faithfulness_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
