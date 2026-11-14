"""Chat request and response schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from policy_rag.schemas.retrieval import (
    Citation,
    ExitReason,
    GradeStatus,
    HopTrace,
    ResponseAction,
    RetrievedChunk,
    RetrievalStrategy,
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="用户提问内容")
    session_id: str | None = Field(default=None, max_length=64, description="会话ID")
    user_id: str = Field(default="anonymous", min_length=1, max_length=64, description="用户标识")
    history: list[ChatMessage] = Field(default_factory=list, description="对话历史")
    policy_context: dict[str, Any] | None = Field(default=None, description="政策版本上下文（如 as_of）")


class LatencyBreakdown(BaseModel):
    guardrail_ms: float = 0.0
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    faithfulness_ms: float = 0.0
    total_ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    action: ResponseAction = ResponseAction.ANSWER
    strategy: RetrievalStrategy = RetrievalStrategy.SINGLE
    user_query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    hop_count: int = 0
    hop_traces: list[HopTrace] = Field(default_factory=list)
    final_grade_status: GradeStatus = GradeStatus.SUFFICIENT
    exit_reason: ExitReason = ExitReason.SINGLE_HOP_DONE
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    known_slots: dict[str, Any] = Field(default_factory=dict)
    clarification_questions: list[str] = Field(default_factory=list)
    refused: bool = False
    error: str | None = None
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class CapabilitiesResponse(BaseModel):
    """Runtime capability & degradation status for white-box diagnostics."""

    status: Literal["ready", "degraded"] = "ready"
    vector_store: str
    clause_graph_enabled: bool = True
    clause_graph_nodes: int = 0
    clause_graph_edges: int = 0
    embedding_backend: str
    reranker_backend: str
    llm_backend: str
    chunk_count: int = 0
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
