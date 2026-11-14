"""Dependency container and core Policy QA Service."""

from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import logging
import time
from typing import Any

from policy_rag.agents.graph import build_graph
from policy_rag.config import Settings, get_settings
from policy_rag.indexing.pipeline import load_and_chunk_knowledge
from policy_rag.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.providers.embeddings import BgeEmbeddingProvider, build_embedding_provider
from policy_rag.providers.fake import FakeLLMProvider, HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.clause_graph import ClauseDependencyGraph
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
    VectorRetriever,
    build_vector_store,
)
from policy_rag.schemas.chat import (
    CapabilitiesResponse,
    ChatRequest,
    ChatResponse,
    LatencyBreakdown,
)
from policy_rag.schemas.retrieval import (
    Citation,
    ExitReason,
    GradeStatus,
    HopTrace,
    ResponseAction,
    RetrievedChunk,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


class PolicyContainer:
    """Dependency injection container holding instantiated singletons."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: RerankProvider | None = None,
        chunks: list[RetrievedChunk] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if llm is not None:
            self.llm = llm
        elif self.settings.llm_configured and not self.settings.is_testing:
            self.llm = OpenAICompatibleLLMProvider(
                api_key=self.settings.active_llm_api_key,
                base_url=self.settings.active_llm_base_url,
                model=self.settings.active_llm_model,
                timeout_seconds=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
        else:
            self.llm = FakeLLMProvider(
                default_response="商城政策规定：普通商品支持7天无理由退换货，商品需保持完好不影响二次销售。"
            )

        self.embedder = embedder or build_embedding_provider(self.settings)
        if reranker is not None:
            self.reranker = reranker
        elif self.settings.is_testing and not self.settings.cross_encoder_force:
            self.reranker = LexicalRerankProvider()
        else:
            self.reranker = CrossEncoderRerankProvider(self.settings)

        self.knowledge_load_error: str | None = None
        if chunks is not None:
            self.chunks = chunks
        else:
            try:
                self.chunks = load_and_chunk_knowledge(self.settings.knowledge_dir)
            except Exception as exc:
                self.knowledge_load_error = str(exc)
                logger.exception(f"Failed to load knowledge directory {self.settings.knowledge_dir}: {exc}")
                self.chunks = []

        # Retrieval components (Qdrant + BM25 + RRF + Cross-Encoder)
        self.bm25 = BM25Retriever(self.chunks)
        self.vector_store = build_vector_store(
            backend=self.settings.vector_store_backend,
            collection=self.settings.qdrant_collection,
            dim=getattr(self.embedder, "dim", self.settings.embedding_dim),
            url=self.settings.qdrant_url,
            path=self.settings.qdrant_path,
        )
        self.vector_retriever = VectorRetriever(self.vector_store, self.embedder)
        self.clause_graph = ClauseDependencyGraph(self.chunks)
        self.hybrid_retriever = HybridRetriever(
            self.bm25,
            self.vector_retriever,
            rrf_k=self.settings.rrf_k,
            clause_graph=self.clause_graph,
        )
        self.document_reranker = DocumentReranker(self.reranker)
        self._vector_lock = asyncio.Lock()
        self.graph = build_graph(
            retriever=self.hybrid_retriever,
            reranker=self.document_reranker,
            llm=self.llm,
            max_hops=self.settings.max_retrieval_hops,
            top_k=self.settings.rerank_top_n,
            parent_expansion=self.settings.parent_expansion_enabled,
            max_parent_chars=self.settings.max_parent_chars,
            max_context_chars=self.settings.max_context_chars,
            default_as_of=self.settings.default_as_of,
        )

        self.qa_service = PolicyQAService(self)

    async def init_vector_index(self) -> None:
        """Encodes all loaded knowledge chunks into in-memory vector store."""
        if not self.chunks or len(self.vector_store) > 0:
            return
        async with self._vector_lock:
            if not self.chunks or len(self.vector_store) > 0:
                return
            texts = [f"[{c.title} - {c.section or '条款'}] {c.content}" for c in self.chunks]
            embeddings = await self.embedder.embed(texts)
            records = [VectorRecord(chunk=c, embedding=emb) for c, emb in zip(self.chunks, embeddings)]
            self.vector_store.upsert(records)

    def get_capabilities(self) -> CapabilitiesResponse:
        """Reports active retrieval/inference backends and any fallback degradations."""
        degradation_reasons: list[str] = []

        if self.knowledge_load_error or not self.chunks:
            degradation_reasons.append("knowledge:load_failed_or_empty")

        if isinstance(self.vector_store, QdrantVectorStore):
            vector_backend = "qdrant"
        else:
            vector_backend = "in_memory"
            degradation_reasons.append("vector_store:in_memory_fallback")

        if isinstance(self.embedder, HashEmbeddingProvider):
            embedding_backend = "hash-fallback"
            degradation_reasons.append("embedding:hash_fallback")
        elif isinstance(self.embedder, BgeEmbeddingProvider):
            if getattr(self.embedder, "_load_error", None) or not self.embedder._should_load():
                embedding_backend = "hash-fallback"
                degradation_reasons.append("embedding:bge_unavailable_fallback")
            else:
                embedding_backend = self.embedder.backend
        else:
            embedding_backend = getattr(self.embedder, "backend", type(self.embedder).__name__)

        if isinstance(self.reranker, LexicalRerankProvider):
            reranker_backend = "lexical-fallback"
            degradation_reasons.append("reranker:lexical_fallback")
        elif isinstance(self.reranker, CrossEncoderRerankProvider):
            if getattr(self.reranker, "_load_error", None) or not self.reranker._should_load():
                reranker_backend = "lexical-fallback"
                degradation_reasons.append("reranker:cross_encoder_unavailable_fallback")
            else:
                reranker_backend = self.reranker.backend
        else:
            reranker_backend = getattr(self.reranker, "backend", type(self.reranker).__name__)

        if isinstance(self.llm, OpenAICompatibleLLMProvider):
            llm_backend = f"openai-compat:{self.llm.model}"
        else:
            llm_backend = "fake-llm"
            degradation_reasons.append("llm:fake_provider")

        degraded = bool(degradation_reasons)
        return CapabilitiesResponse(
            status="degraded" if degraded else "ready",
            vector_store=vector_backend,
            clause_graph_enabled=self.clause_graph.edge_count > 0,
            clause_graph_nodes=self.clause_graph.node_count,
            clause_graph_edges=self.clause_graph.edge_count,
            embedding_backend=embedding_backend,
            reranker_backend=reranker_backend,
            llm_backend=llm_backend,
            chunk_count=len(self.chunks),
            degraded=degraded,
            degradation_reasons=degradation_reasons,
        )


class PolicyQAService:
    """End-to-end orchestration service for policy Q&A."""

    def __init__(self, container: PolicyContainer) -> None:
        self.container = container
        self.settings = container.settings
        self.llm = container.llm
        self.sessions: dict[str, list[dict[str, str]]] = {}

    async def _prepare_execution(self, request: ChatRequest) -> tuple[str, dict[str, Any]]:
        query = request.message.strip()
        await self.container.init_vector_index()

        history: list[dict[str, str]] = []
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]
        elif request.session_id:
            history = list(self.sessions.get(request.session_id, []))

        metadata_filter = infer_retrieval_filter(
            query,
            default_as_of=self.settings.default_as_of,
            policy_context=request.policy_context,
            history=history,
        )

        initial_state: dict[str, Any] = {
            "user_query": query,
            "conversation_history": history,
            "policy_context": request.policy_context or {},
            "metadata_filter": metadata_filter,
        }
        return query, initial_state

    def _finalize_response(
        self,
        query: str,
        request: ChatRequest,
        final_state: dict[str, Any],
        t0: float,
    ) -> ChatResponse:
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency = LatencyBreakdown(
            guardrail_ms=round(float(final_state.get("guardrail_ms", 0.0)), 2),
            routing_ms=round(float(final_state.get("routing_ms", 0.0)), 2),
            retrieval_ms=round(float(final_state.get("retrieval_ms", 0.0)), 2),
            rerank_ms=round(float(final_state.get("rerank_ms", 0.0)), 2),
            llm_ms=round(float(final_state.get("llm_ms", 0.0)), 2),
            faithfulness_ms=round(float(final_state.get("faithfulness_ms", 0.0)), 2),
            total_ms=total_ms,
        )

        answer_text = final_state.get("final_answer", "")
        strategy = final_state.get("strategy", RetrievalStrategy.SINGLE)
        action = final_state.get("action", ResponseAction.ANSWER)
        refused = bool(final_state.get("refused", False))
        if refused:
            action = ResponseAction.REFUSE
        exit_reason = final_state.get("exit_reason", ExitReason.SINGLE_HOP_DONE)
        final_grade_status = final_state.get("final_grade_status", GradeStatus.SUFFICIENT)

        if request.session_id and not refused and answer_text:
            session_history = self.sessions.setdefault(request.session_id, [])
            session_history.append({"role": "user", "content": query})
            session_history.append({"role": "assistant", "content": answer_text})
            if len(session_history) > 12:
                self.sessions[request.session_id] = session_history[-12:]

        return ChatResponse(
            answer=answer_text,
            action=action,
            strategy=strategy,
            user_query=query,
            rewritten_queries=final_state.get("rewritten_queries", []),
            hop_count=final_state.get(
                "hop_count",
                1 if strategy in {RetrievalStrategy.SINGLE, RetrievalStrategy.MULTI} else 0,
            ),
            hop_traces=final_state.get("hop_traces", []),
            final_grade_status=final_grade_status,
            exit_reason=exit_reason,
            evidence=final_state.get("evidence", []),
            citations=final_state.get("citations", []),
            missing_slots=final_state.get("missing_slots", []),
            known_slots=final_state.get("known_slots", {}),
            clarification_questions=final_state.get("clarification_questions", []),
            refused=refused,
            error=final_state.get("error"),
            latency=latency,
        )

    async def answer(self, request: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        query, initial_state = await self._prepare_execution(request)
        final_state = await self.container.graph.ainvoke(initial_state)
        return self._finalize_response(query, request, final_state, t0)

    async def stream_answer(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Streams LangGraph node transitions, answer token slices, and final verification payload."""
        t0 = time.perf_counter()
        query, initial_state = await self._prepare_execution(request)
        accumulated_state: dict[str, Any] = dict(initial_state)

        yield {
            "event": "status",
            "data": {"stage": "started", "user_query": query},
        }

        async for update in self.container.graph.astream(initial_state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_output in update.items():
                if not isinstance(node_output, dict):
                    continue
                accumulated_state.update(node_output)
                payload: dict[str, Any] = {"node": node_name}
                for key in (
                    "strategy",
                    "action",
                    "hop_count",
                    "final_grade_status",
                    "exit_reason",
                    "rewritten_queries",
                    "missing_slots",
                    "guardrail_ms",
                    "routing_ms",
                    "retrieval_ms",
                    "rerank_ms",
                    "llm_ms",
                    "faithfulness_ms",
                ):
                    if key in node_output and node_output[key] is not None:
                        val = node_output[key]
                        payload[key] = val.value if hasattr(val, "value") else val
                if "evidence" in node_output and node_output["evidence"] is not None:
                    payload["evidence_count"] = len(node_output["evidence"])
                if "citations" in node_output and node_output["citations"] is not None:
                    payload["citations_count"] = len(node_output["citations"])
                yield {"event": "node_update", "data": payload}

        response = self._finalize_response(query, request, accumulated_state, t0)
        answer_text = response.answer or ""
        chunk_size = 24
        for idx in range(0, len(answer_text), chunk_size):
            yield {"event": "token", "data": {"delta": answer_text[idx : idx + chunk_size]}}

        yield {"event": "done", "data": response.model_dump(mode="json")}
