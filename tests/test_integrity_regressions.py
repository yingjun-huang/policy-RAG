"""Regression tests for observed logic failures; no live APIs or gold-driven generation."""
import pytest

from policy_rag.agents.graph import build_graph
from policy_rag.agents.loop import _parse_bool
from policy_rag.agents.nodes import faithfulness_node, generate_node, self_rag_node, should_run_faithfulness
from policy_rag.agents.router import route_query
from policy_rag.providers.base import LLMResponse
from policy_rag.providers.fake import FakeLLMProvider, HashEmbeddingProvider, LexicalRerankProvider
from policy_rag.retrieval.bm25 import BM25Retriever
from policy_rag.retrieval.filters import infer_retrieval_filter
from policy_rag.retrieval.hybrid import HybridRetriever
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.retrieval.vector import InMemoryVectorStore, VectorRetriever
from policy_rag.schemas.chat import ChatResponse
from policy_rag.schemas.retrieval import ExitReason, GradeStatus, ResponseAction, RetrievedChunk, RetrievalStrategy


def _state():
    chunk = RetrievedChunk(document_id='doc', chunk_id='c1', title='规则', content='七天内可申请，保修三年。')
    return {'user_query':'能退吗', 'strategy':RetrievalStrategy.SINGLE, 'action':ResponseAction.ANSWER,
            'evidence':[chunk], 'final_grade_status':GradeStatus.SUFFICIENT, 'final_answer':'可在7天内申请。'}


@pytest.mark.parametrize('answer', ['可在7天内申请。', '保修3年。', '三年保修。', '七个自然日内办理。', '30分钟后到账。'])
def test_temporal_claim_without_money_is_high_risk(answer):
    assert should_run_faithfulness({**_state(), 'final_answer':answer})[0]


@pytest.mark.parametrize('value', [None, 'maybe', {}, 2])
def test_unknown_boolean_is_unavailable(value):
    with pytest.raises(ValueError):
        _parse_bool(value)


@pytest.mark.asyncio
async def test_semantic_refusal_is_not_overridden_by_question_particle():
    strategy, _ = await route_query('能帮我把这单撤掉吗？', FakeLLMProvider(default_response='{"strategy":"refuse"}'))
    assert strategy == RetrievalStrategy.REFUSE


@pytest.mark.asyncio
async def test_llm_refusal_has_customer_facing_text():
    hybrid = HybridRetriever(BM25Retriever([]), VectorRetriever(InMemoryVectorStore(), HashEmbeddingProvider()))
    graph = build_graph(hybrid, DocumentReranker(LexicalRerankProvider()), FakeLLMProvider(default_response='{"strategy":"refuse"}'))
    result = await graph.ainvoke({'user_query':'请替我申请退款'})
    assert result['refused'] and result['final_answer'].strip()


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['NOT_JSON', '{}', '{"answer":"可以退","action":"wrong"}', '{"answer":"   "}'])
async def test_invalid_generation_does_not_publish_raw_output_with_citations(raw):
    result = await generate_node(_state(), llm=FakeLLMProvider(default_response=raw))
    assert result['action'] == ResponseAction.ABSTAIN
    assert result['citations'] == []
    assert result['error'] == 'generation_invalid_response'


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['NOT_JSON', '{}', '{"supported":"maybe"}'])
async def test_failed_checker_is_not_counted_as_checked(raw):
    result = await faithfulness_node(_state(), llm=FakeLLMProvider(default_response=raw), risk_gate_enabled=False)
    assert result['faithfulness_checked'] is False
    assert result['error'] == 'faithfulness_unavailable'


class _SequentialLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return LLMResponse(content=next(self.replies))


@pytest.mark.asyncio
async def test_failed_correction_withdraws_rejected_answer():
    llm = _SequentialLLM(['{"supported":false}', '{}'])
    result = await faithfulness_node(_state(), llm=llm, risk_gate_enabled=False)
    assert result['action'] == ResponseAction.ABSTAIN
    assert result['faithfulness_checked'] and result['faithfulness_triggered']
    assert result['error'] == 'self_correction_failed'
    assert result['pre_correction_answer'] == _state()['final_answer']


@pytest.mark.asyncio
async def test_correction_synchronizes_action_slots_exit_and_prior_action():
    state = {**_state(), 'action':ResponseAction.ABSTAIN, 'exit_reason':ExitReason.ABSTAINED,
             'missing_slots':['amount'], 'clarification_questions':['金额多少？']}
    llm = _SequentialLLM(['{"supported":false}', '{"action":"answer","answer":"有据答案","citation_chunk_ids":["c1"]}'])
    result = await faithfulness_node(state, llm=llm, risk_gate_enabled=False)
    assert result['action'] == ResponseAction.ANSWER
    assert result['missing_slots'] == result['clarification_questions'] == []
    assert result['exit_reason'] == ExitReason.SINGLE_HOP_DONE
    response = ChatResponse(answer=result['final_answer'], user_query='q', pre_correction_action=result['pre_correction_action'])
    assert response.model_dump(mode='json')['pre_correction_action'] == 'abstain'


@pytest.mark.asyncio
async def test_faithfulness_sees_prior_user_facts():
    state = {**_state(), 'conversation_history':[{'role':'user','content':'实付45元，重量10.2kg。'}]}
    llm = _SequentialLLM(['{"supported":true}'])
    await faithfulness_node(state, llm=llm, risk_gate_enabled=False)
    assert '45元' in llm.messages[0][1]['content'] and '10.2kg' in llm.messages[0][1]['content']


@pytest.mark.asyncio
async def test_multihop_grading_sees_prior_user_facts():
    state = {**_state(), 'user_query':'上海，不是PLUS。',
             'conversation_history':[{'role':'user','content':'实付45元，重量10.2kg。'}]}
    class Retriever:
        async def retrieve(self, query, **kwargs):
            return state['evidence']
    llm = _SequentialLLM(['{"sufficient":true}'])
    await self_rag_node(state, retriever=Retriever(), reranker=DocumentReranker(LexicalRerankProvider()), llm=llm)
    assert '45元' in llm.messages[0][1]['content'] and '10.2kg' in llm.messages[0][1]['content']


def test_assistant_historical_claim_does_not_unlock_archives():
    result = infer_retrieval_filter('今天70元的自营订单怎么计费？', history=[{'role':'assistant','content':'2023年旧版是99元门槛。'}])
    assert result.include_archived is False
