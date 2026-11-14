"""LangGraph shared state definition."""

from __future__ import annotations

from typing import Any, TypedDict
from policy_rag.schemas.chat import LatencyBreakdown
from policy_rag.schemas.retrieval import (
    Citation,
    ExitReason,
    GradeStatus,
    HopTrace,
    MetadataFilter,
    ResponseAction,
    RetrievedChunk,
    RetrievalStrategy,
)


class AgentState(TypedDict, total=False):
    """Global state flowing through all LangGraph nodes."""

    user_query: str
    conversation_history: list[dict[str, str]]
    policy_context: dict[str, Any]
    metadata_filter: MetadataFilter | None
    strategy: RetrievalStrategy
    action: ResponseAction
    rewritten_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]
    hop_count: int
    hop_traces: list[HopTrace]
    final_grade_status: GradeStatus
    exit_reason: ExitReason
    missing_aspect: str
    missing_slots: list[str]
    known_slots: dict[str, Any]
    clarification_questions: list[str]
    evidence: list[RetrievedChunk]
    final_answer: str
    faithfulness_triggered: bool
    pre_correction_answer: str | None
    pre_correction_action: ResponseAction | None
    citations: list[Citation]
    refused: bool
    error: str | None
    guardrail_ms: float
    routing_ms: float
    retrieval_ms: float
    rerank_ms: float
    llm_ms: float
    faithfulness_ms: float
    latency: LatencyBreakdown
