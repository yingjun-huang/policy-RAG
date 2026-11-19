"""Business decision regressions; simulated model outputs, never a model accuracy score."""
import json
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from policy_rag.agents.contracts import AnswerPayload, EvidenceGrade
from policy_rag.agents.graph import build_graph
from policy_rag.agents.loop import run_self_rag_loop
from policy_rag.agents.nodes import generate_node, faithfulness_node, retrieve_single_node, _map_citations
from policy_rag.agents.router import route_query
from policy_rag.api.main import create_app
from policy_rag.config import Settings
from policy_rag.providers.base import LLMResponse
from policy_rag.retrieval.filters import infer_retrieval_filter, matches_metadata_filter
from policy_rag.schemas.chat import ChatRequest, ChatResponse
from policy_rag.schemas.retrieval import RetrievedChunk, MetadataFilter, GradeStatus, ExitReason, ResponseAction, RetrievalStrategy
from policy_rag.service import PolicyContainer

def chunk(cid="c1", **metadata):
    return RetrievedChunk(document_id="policy", chunk_id=cid, title="退货条件",
                          content="适用退货条件须根据用户签收日期判断。", metadata=metadata)

def state(**updates):
    result = dict(user_query="普通商品退货条件？", strategy=RetrievalStrategy.SINGLE,
                  action=ResponseAction.ANSWER, evidence=[chunk()], final_grade_status=GradeStatus.SUFFICIENT,
                  final_answer="须在7天内申请。", exit_reason=ExitReason.SINGLE_HOP_DONE)
    result.update(updates)
    return result

class Fake:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
    async def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.replies:
            raise RuntimeError("Injected provider failure")
        return LLMResponse(content=self.replies.pop(0))

class Retriever:
    def __init__(self, mode="same"):
        self.calls, self.mode = [], mode
    async def retrieve(self, query, **kwargs):
        self.calls.append(query)
        if self.mode == "empty":
            return []
        return [chunk("c1" if self.mode == "same" else f"c{len(self.calls)}")]

class Reranker:
    async def rerank(self, query, chunks, **kwargs):
        return chunks

SUFFICIENT = '{"sufficient":true,"missing_type":"none"}'
POLICY_GAP = '{"sufficient":false,"missing_type":"policy_evidence","missing_aspect":"缺专项退货细则"}'
def user_gap(kind="user_facts"):
    return json.dumps(dict(sufficient=False, missing_type=kind, missing_aspect="签收日期未知",
                           missing_slots=["signed_at"], clarification_questions=["请问哪一天签收？"]), ensure_ascii=False)

@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected", [("帮我取消订单", RetrievalStrategy.REFUSE), ("你好", RetrievalStrategy.DIRECT)])
async def test_confirmed_fast_paths_do_not_need_model(query, expected):
    llm = Fake()
    assert (await route_query(query, llm))[0] == expected
    assert llm.calls == []

@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"strategy":"direct"}', "NOT_JSON"])
async def test_policy_query_cannot_bypass_evidence(response):
    r = Retriever()
    llm = Fake(response, SUFFICIENT, '{"answer":"符合条件时可申请退货。","citation_chunk_ids":["c1"]}')
    result = await build_graph(r, Reranker(), llm).ainvoke({"user_query":"普通商品退货条件是什么？"})
    assert result["strategy"] == RetrievalStrategy.SINGLE
    assert len(r.calls) == 1
    assert result["final_answer"] == "符合条件时可申请退货。"

@pytest.mark.asyncio
async def test_single_runs_real_grader_and_does_not_assume_sufficient():
    llm = Fake(POLICY_GAP)
    result = await retrieve_single_node(state(), retriever=Retriever(), reranker=Reranker(), llm=llm,
                                        parent_expansion=False)
    assert len(llm.calls) == 1
    assert result["final_grade_status"] == GradeStatus.INSUFFICIENT
    assert result["exit_reason"] == ExitReason.MAX_HOPS_REACHED

@pytest.mark.asyncio
async def test_no_evidence_does_not_generate():
    llm = Fake()
    result = await generate_node(state(evidence=[]), llm=llm)
    assert result["action"] == ResponseAction.ABSTAIN and not llm.calls

@pytest.mark.asyncio
async def test_sufficient_multi_ends_in_first_round():
    llm, r = Fake(SUFFICIENT), Retriever()
    result = await run_self_rag_loop("条件？", r, Reranker(), llm, parent_expansion=False)
    assert result["hop_count"] == 1 and len(r.calls) == len(llm.calls) == 1
    assert result["exit_reason"] == ExitReason.SUFFICIENT_EVIDENCE

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user_facts", "both"])
@pytest.mark.parametrize("strategy", ["single", "multi"])
async def test_blocking_user_facts_clarify_without_more_generation(kind, strategy):
    llm, r = Fake(json.dumps({"strategy": strategy}), user_gap(kind)), Retriever("new")
    result = await build_graph(r, Reranker(), llm, faithfulness_risk_gate_enabled=False).ainvoke(
        {"user_query":"普通商品现在能退吗？"})
    assert result["action"] == ResponseAction.CLARIFY
    assert result["exit_reason"] == ExitReason.CLARIFICATION_NEEDED
    assert result["missing_type"] == kind
    assert result["missing_slots"] == ["signed_at"]
    assert result["clarification_questions"] == ["请问哪一天签收？"]
    assert "请问哪一天签收" in result["final_answer"]
    assert len(r.calls) == 1 and len(llm.calls) == 2
    assert not result["faithfulness_checked"]

@pytest.mark.asyncio
async def test_policy_gap_can_retrieve_second_round():
    llm = Fake(POLICY_GAP, '{"queries":["专项退货细则"]}', SUFFICIENT)
    r = Retriever("new")
    result = await run_self_rag_loop("条件？", r, Reranker(), llm, parent_expansion=False)
    assert result["hop_count"] == 2 and len(r.calls) == 2
    assert result["final_grade_status"] == GradeStatus.SUFFICIENT

@pytest.mark.asyncio
async def test_no_new_evidence_stops_before_another_grade():
    llm = Fake(POLICY_GAP, '{"queries":["专项细则"]}')
    result = await run_self_rag_loop("条件？", Retriever(), Reranker(), llm, parent_expansion=False)
    assert result["exit_reason"] == ExitReason.NO_NEW_EVIDENCE
    assert result["final_grade_status"] == GradeStatus.INSUFFICIENT and len(llm.calls) == 2

@pytest.mark.asyncio
async def test_two_round_limit_does_not_become_sufficient():
    llm = Fake(POLICY_GAP, '{"queries":["专项细则"]}', POLICY_GAP)
    result = await run_self_rag_loop("条件？", Retriever("new"), Reranker(), llm, parent_expansion=False)
    assert result["hop_count"] == 2 and len(llm.calls) == 3
    assert result["exit_reason"] == ExitReason.MAX_HOPS_REACHED
    assert result["final_grade_status"] == GradeStatus.INSUFFICIENT

@pytest.mark.asyncio
@pytest.mark.parametrize("reply, expected_second_queries", [
    ('{"queries":"专项细则"}', ["条件？ 缺专项退货细则"]),
    ('{"queries":["一","二","三","四"]}', ["条件？ 缺专项退货细则"]),
    ('{"queries":["专项细则","补充条款"]}', ["专项细则", "补充条款"]),
])
async def test_rewrite_type_and_query_budget(reply, expected_second_queries):
    llm, r = Fake(POLICY_GAP, reply, SUFFICIENT), Retriever("new")
    result = await run_self_rag_loop("条件？", r, Reranker(), llm, parent_expansion=False)
    assert r.calls[1:] == expected_second_queries
    assert result["rewritten_queries"] == expected_second_queries
    assert len(r.calls) <= 3

@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["NOT_JSON", '{"sufficient":false,"missing_aspect":"缺条款"}',
    '{"sufficient":false,"missing_type":"user_facts","missing_aspect":"缺签收日期"}',
    '{"sufficient":true,"missing_type":"user_facts"}'])
async def test_invalid_grade_abstains_without_being_revived(reply):
    r, llm = Retriever(), Fake('{"strategy":"multi"}', reply)
    result = await build_graph(r, Reranker(), llm, faithfulness_risk_gate_enabled=False).ainvoke(
        {"user_query":"普通商品退货条件是什么？"})
    assert result["action"] == ResponseAction.ABSTAIN
    assert result["final_grade_status"] == GradeStatus.GRADER_UNAVAILABLE
    assert result["exit_reason"] == ExitReason.GRADER_UNAVAILABLE
    assert result["error"] == "grader_unavailable"
    assert len(r.calls) == 1 and len(llm.calls) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["NOT_JSON", '{"action":"clarify","answer":"需补充条件。"}', None])
async def test_invalid_generation_withdraws_answer(reply):
    llm = Fake() if reply is None else Fake(reply)
    result = await generate_node(state(), llm=llm)
    assert result["action"] == ResponseAction.ABSTAIN
    assert result["error"] == "generation_invalid_response" and result["citations"] == []

@pytest.mark.asyncio
@pytest.mark.parametrize("risk_gate", [False, True])
async def test_generation_error_never_restarts_answer(risk_gate):
    llm = Fake('{"supported":false}', '{"answer":"被重新发布的结论"}')
    result = await faithfulness_node(state(error="generation_invalid_response", action=ResponseAction.ABSTAIN),
                                     llm=llm, risk_gate_enabled=risk_gate)
    assert not llm.calls and not result["faithfulness_checked"]
    assert "final_answer" not in result

def test_missing_or_invalid_citations_remain_empty():
    assert _map_citations([], [chunk()]) == []
    assert _map_citations(["invented"], [chunk()]) == []
    assert [c.chunk_id for c in _map_citations(["c1", "invented", "c1"], [chunk()])] == ["c1"]

@pytest.mark.asyncio
async def test_checker_failure_is_not_success():
    result = await faithfulness_node(state(), llm=Fake("NOT_JSON"), risk_gate_enabled=False)
    assert not result["faithfulness_checked"] and result["error"] == "faithfulness_unavailable"

@pytest.mark.asyncio
@pytest.mark.parametrize("correction", ["NOT_JSON", '{"action":"clarify","answer":"需补充条件。"}'])
async def test_invalid_correction_withdraws_original_answer(correction):
    llm = Fake('{"supported":false}', correction)
    result = await faithfulness_node(state(), llm=llm, risk_gate_enabled=False)
    assert result["action"] == ResponseAction.ABSTAIN
    assert result["error"] == "self_correction_failed" and result["citations"] == []
    assert len(llm.calls) == 2 and result["pre_correction_answer"] == "须在7天内申请。"

@pytest.mark.asyncio
async def test_successful_correction_has_one_attempt():
    llm = Fake('{"supported":false}', '{"answer":"满足条件后申请。","citation_chunk_ids":["c1"]}')
    result = await faithfulness_node(state(), llm=llm, risk_gate_enabled=False)
    assert result["final_answer"] == "满足条件后申请。" and result["faithfulness_triggered"]
    assert len(llm.calls) == 2

@pytest.mark.parametrize("date, old_allowed, new_allowed", [
    ("2023-01-01", True, False), ("2023-12-31", True, False), ("2024-01-01", False, True)])
def test_as_of_selects_only_effective_version(date, old_allowed, new_allowed):
    mf = infer_retrieval_filter("适用条件", policy_context={"as_of": date})
    old = chunk(document_status="archived", effective_from="2022-01-01", effective_to="2024-01-01")
    new = chunk(document_status="current", effective_from="2024-01-01")
    assert matches_metadata_filter(old, mf) == old_allowed
    assert matches_metadata_filter(new, mf) == new_allowed

def test_comparison_keeps_both_versions_and_context_metadata():
    request = ChatRequest(message="新旧规则对比", policy_context={"namespace":"jd", "corpus_revision":"frozen"})
    assert request.policy_context["namespace"] == "jd"
    assert request.policy_context["corpus_revision"] == "frozen"
    mf = infer_retrieval_filter(request.message, policy_context=request.policy_context)
    assert mf.include_archived and mf.as_of is None
    assert matches_metadata_filter(chunk(document_status="archived", effective_to="2024-01-01"), mf)

@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"message":"   "}, {"message":"政策", "policy_context":{"as_of":"2023-02-30"}},
    {"message":"政策", "policy_context":{"as_of":123}},
    {"message":"政策", "policy_context":{"merchant_types":"自营"}},
    {"message":"政策", "history":[{"role":"user", "content":"x"}]*13}])
async def test_invalid_external_fields_return_422(client, payload):
    assert (await client.post("/api/v1/chat", json=payload)).status_code == 422

class ClarificationFlow:
    def __init__(self):
        self.calls = []
    async def generate(self, messages, **kwargs):
        self.calls.append(messages[0]["content"])
        if "检索策略分类器" in messages[0]["content"]:
            return LLMResponse(content='{"strategy":"multi"}')
        if "政策证据评审员" in messages[0]["content"]:
            return LLMResponse(content=user_gap())
        raise AssertionError("Pending facts must not cause generation, rewrite or correction calls")

@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_clarification_reaches_http_and_sse(stream):
    llm = ClarificationFlow()
    container = PolicyContainer(settings=Settings(_env_file=None, app_env="testing"), llm=llm, chunks=[chunk()])
    app = create_app(container)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        path = "/api/v1/chat/stream" if stream else "/api/v1/chat"
        response = await client.post(path, json={"message":"普通商品现在能退吗？"})
        assert response.status_code == 200
        if stream:
            events = []
            for block in response.text.replace("\r\n", "\n").split("\n\n"):
                lines = block.splitlines()
                event = next((line[6:].strip() for line in lines if line.startswith("event:")), None)
                data = next((line[5:].strip() for line in lines if line.startswith("data:")), None)
                if event and data:
                    events.append((event, json.loads(data)))
            result = next(data for event, data in events if event == "done")
            assert "".join(data["delta"] for event, data in events if event == "token") == result["answer"]
        else:
            result = response.json()
        parsed = ChatResponse.model_validate(result)
        assert parsed.action == ResponseAction.CLARIFY and parsed.hop_count == 1
        assert parsed.missing_type == "user_facts"
        assert parsed.clarification_questions == ["请问哪一天签收？"] and parsed.missing_slots == ["signed_at"]
        assert not parsed.faithfulness_checked and len(llm.calls) == 2

def test_invalid_judge_output_does_not_improve_score():
    from eval.judges.answer_judge import AnswerJudgment
    from eval.judges.specialist_metrics import summarize_judgments
    rows = [{"judgment": AnswerJudgment(verdict=verdict, score=score)}
            for verdict, score in [("correct", 1), ("partial", .5), ("incorrect", 0)]]
    rows.append({"judgment": AnswerJudgment(verdict="correct", score=1, parse_fallback=True)})
    result = summarize_judgments(rows, "offline")
    assert result["evaluated"] == 3 and result["weighted_accuracy"] == 50
