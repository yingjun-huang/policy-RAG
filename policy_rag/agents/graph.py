"""LangGraph state machine wiring all policy QA nodes together."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from policy_rag.agents.nodes import (
    faithfulness_node,
    generate_node,
    guardrail_node,
    refuse_node,
    retrieve_single_node,
    router_node,
    self_rag_node,
)
from policy_rag.agents.state import AgentState
from policy_rag.providers.base import LLMProvider
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.schemas.retrieval import RetrievalStrategy


def _after_guardrail(state: AgentState) -> str:
    if state.get("refused"):
        return "refuse"
    return "router"


def _route_strategy(state: AgentState) -> str:
    strategy = state.get("strategy")
    if strategy == RetrievalStrategy.REFUSE:
        return "refuse"
    if strategy == RetrievalStrategy.DIRECT:
        return "generate"
    if strategy == RetrievalStrategy.MULTI:
        return "self_rag"
    return "retrieve_single"


def _after_generate(state: AgentState) -> str:
    if state.get("strategy") == RetrievalStrategy.DIRECT:
        return END
    return "faithfulness"


def build_graph(
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    llm: LLMProvider,
    max_hops: int = 2,
    top_k: int = 3,
    *,
    parent_expansion: bool = True,
    max_parent_chars: int = 1500,
    max_context_chars: int = 4500,
    default_as_of: str = "2024-06-01T12:00:00+08:00",
    faithfulness_risk_gate_enabled: bool = True,
) -> CompiledStateGraph:
    """Constructs and compiles the full LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Register all processing nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node(
        "router",
        partial(router_node, llm=llm, default_as_of=default_as_of),
    )
    workflow.add_node(
        "retrieve_single",
        partial(
            retrieve_single_node,
            retriever=retriever,
            reranker=reranker,
            top_k=top_k,
            parent_expansion=parent_expansion,
            max_parent_chars=max_parent_chars,
            max_context_chars=max_context_chars,
            default_as_of=default_as_of,
        ),
    )
    workflow.add_node(
        "self_rag",
        partial(
            self_rag_node,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            max_hops=max_hops,
            top_k=top_k,
            parent_expansion=parent_expansion,
            max_parent_chars=max_parent_chars,
            max_context_chars=max_context_chars,
            default_as_of=default_as_of,
        ),
    )
    workflow.add_node("generate", partial(generate_node, llm=llm))
    workflow.add_node("refuse", refuse_node)
    workflow.add_node(
        "faithfulness",
        partial(faithfulness_node, llm=llm, risk_gate_enabled=faithfulness_risk_gate_enabled),
    )

    # Flow edges
    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        _after_guardrail,
        {"refuse": "refuse", "router": "router"},
    )

    workflow.add_conditional_edges(
        "router",
        _route_strategy,
        {
            "refuse": "refuse",
            "generate": "generate",
            "retrieve_single": "retrieve_single",
            "self_rag": "self_rag",
        },
    )

    workflow.add_edge("retrieve_single", "generate")
    workflow.add_edge("self_rag", "generate")

    workflow.add_conditional_edges(
        "generate",
        _after_generate,
        {END: END, "faithfulness": "faithfulness"},
    )

    workflow.add_edge("faithfulness", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()
