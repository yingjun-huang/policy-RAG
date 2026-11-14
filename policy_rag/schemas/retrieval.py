"""Core retrieval and chunk schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    DIRECT = "direct"
    SINGLE = "single"
    MULTI = "multi"
    REFUSE = "refuse"


class ResponseAction(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    REFUSE = "refuse"


class GradeStatus(str, Enum):
    SUFFICIENT = "sufficient"                 # 证据充分，足以直接回答
    INSUFFICIENT = "insufficient"             # 证据不足，存在关键缺失条件
    GRADER_UNAVAILABLE = "grader_unavailable" # 评审模型超时、不可用或输出非法 JSON


class ExitReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"   # 证据充分正常退出
    NO_NEW_EVIDENCE = "no_new_evidence"           # 该轮未检出增量新证据提前终止
    MAX_HOPS_REACHED = "max_hops_reached"         # 达到最大配置轮次终止
    SINGLE_HOP_DONE = "single_hop_done"           # 单轮检索完成
    DIRECT_ANSWER = "direct_answer"               # 简单问答/无需检索直答
    GUARDRAIL_REFUSED = "guardrail_refused"       # 安全拦截或业务写操作越界拒答
    CLARIFICATION_NEEDED = "clarification_needed" # 关键业务条件缺失，追问澄清
    NO_EVIDENCE = "no_evidence"                   # 未检出适用证据，拒答/说明未覆盖
    ABSTAINED = "abstained"                       # 知识库未覆盖关键条款，保守拒答


class MetadataFilter(BaseModel):
    """Shared pre-retrieval / pre-fusion metadata filter across BM25 and Qdrant Dense routes."""

    as_of: str | None = None
    include_archived: bool = False
    doc_groups: list[str] | None = None
    policy_families: list[str] | None = None
    merchant_types: list[str] | None = None


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    title: str
    source: str = "knowledge"
    section: str | None = None
    parent_id: str | None = None
    score: float = 0.0
    rank: int | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    section: str | None = None


class HopTrace(BaseModel):
    hop_index: int
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    new_chunk_ids: list[str] = Field(default_factory=list)
    grade_status: GradeStatus
    missing_aspect: str = ""
    rationale: str = ""
