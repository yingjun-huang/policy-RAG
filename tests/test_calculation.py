"""External-boundary and integration regressions, with simulated model responses only."""
import json
from decimal import Decimal
import pytest

from policy_rag.calculation import CalculationError, CalculationRequest, evaluate_expression, resolve_calculations
from policy_rag.agents.nodes import generate_node, faithfulness_node
from policy_rag.agents.router import route_query
from policy_rag.providers.base import LLMResponse
from policy_rag.providers.fake import FakeLLMProvider
from policy_rag.schemas.retrieval import RetrievedChunk, RetrievalStrategy, ResponseAction, GradeStatus
from policy_rag.schemas.chat import ChatResponse


@pytest.mark.parametrize("expression, expected", [
    ("0.1+0.2", "0.3"), ("10+ceil(max(3.4-1,0))*3", "19"),
    ("6+ceil(10.2-5)*1", "12"), ("3000*0.0005*(20-0-0)", "30"),
    ("ceil(5.0001-5)", "1"), ("ceil(5-5)", "0"),
    ("200*0.3", "60"), ("200*0.5", "100"), ("5/20*100", "25"),
    ("min(max(72*0.3,10),500)", "21.6"), ("max(43*10,1000)+43", "1043"),
    ("max(499-399-0,0)", "100"), ("max(35-90,0)", "0"),
    ("-2+3", "1"), ("ceil(-1.2)", "-1"),
])
def test_decimal_arithmetic(expression, expected):
    assert evaluate_expression(expression) == Decimal(expected)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('whoami')", "(1).__class__", "2**100000", "1/0",
    "1//2", "[1,2]", "True+1", "min(x=1)", "sum([1,2])", "ceil(1,2)",
    "1000000000000*2", "1e100", "1+"*80+"1", "nan", "Decimal('1')",
])
def test_untrusted_expression_is_rejected(expression):
    with pytest.raises(CalculationError):
        evaluate_expression(expression)


def evidence():
    return [RetrievedChunk(document_id="shipping", chunk_id="c1", title="附加费",
                           content="首重1kg内10元，超过部分向上取整，每kg3元。")]


def plan(expression="10+ceil(max(3.4-1,0))*3", ident="freight", refs=None):
    return CalculationRequest(id=ident, expression=expression, evidence_chunk_ids=refs or ["c1"])


def test_expression_trace_and_repeated_placeholder():
    answer, trace = resolve_calculations("附加费[[calc:freight]]元，总费[[calc:freight]]元。", [plan()], evidence())
    assert answer == "附加费19.00元，总费19.00元。"
    assert trace[0]["value"] == "19" and trace[0]["evidence_chunk_ids"] == ["c1"]


def test_range_preserves_two_endpoints():
    answer, trace = resolve_calculations("[[calc:low]]至[[calc:high]]元，具体值双方确认。",
        [plan("200*0.3", "low"), plan("200*0.5", "high")], evidence())
    assert answer == "60.00至100.00元，具体值双方确认。" and len(trace) == 2


@pytest.mark.parametrize("answer, requests", [
    ("[[calc:freight]]", []), ("19元", [plan()]),
    ("[[calc:freight]]", [plan(),plan()]),
    ("[[calc:freight]]", [plan(refs=["missing"])]),
    ("[[calc:wrong id]]", []),
])
def test_missing_duplicate_unused_or_unavailable_references_fail(answer, requests):
    with pytest.raises(CalculationError):
        resolve_calculations(answer, requests, evidence())


def state():
    return {"user_query":"3.4kg，首重10元，每超出1kg收3元，怎么算？", "strategy":RetrievalStrategy.SINGLE,
            "evidence":evidence(), "final_grade_status":GradeStatus.SUFFICIENT}


@pytest.mark.asyncio
async def test_generation_uses_computed_amount_and_evidence_citation():
    payload = {"action":"answer", "answer":"附加费[[calc:freight]]元。",
               "calculations":[plan().model_dump()], "citation_chunk_ids":[]}
    result = await generate_node(state(), llm=FakeLLMProvider(default_response=json.dumps(payload)))
    assert result["final_answer"] == "附加费19.00元。"
    assert result["calculation_trace"][0]["rendered"] == "19.00"
    assert result["citations"][0].chunk_id == "c1"
    response = ChatResponse(user_query="q", answer=result["final_answer"], calculation_trace=result["calculation_trace"])
    assert response.model_dump()["calculation_trace"]


@pytest.mark.asyncio
async def test_invalid_calculation_withdraws_answer():
    payload = {"action":"answer", "answer":"退款[[calc:freight]]元。",
               "calculations":[plan("1/0").model_dump()]}
    result = await generate_node(state(), llm=FakeLLMProvider(default_response=json.dumps(payload)))
    assert result["action"] == ResponseAction.ABSTAIN
    assert result["citations"] == [] and result["error"] == "generation_invalid_response"


class Sequence:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def generate(self, messages, **kwargs):
        return LLMResponse(content=next(self.responses))


@pytest.mark.asyncio
async def test_faithfulness_correction_also_uses_calculator():
    payload = {"action":"answer", "answer":"附加费[[calc:freight]]元。", "calculations":[plan().model_dump()]}
    result = await faithfulness_node({**state(),"final_answer":"附加费16元。","action":ResponseAction.ANSWER},
        llm=Sequence(['{"supported":false}',json.dumps(payload)]), risk_gate_enabled=False)
    assert result["final_answer"] == "附加费19.00元。" and result["calculation_trace"]
    assert result["pre_correction_answer"] == "附加费16元。"


@pytest.mark.asyncio
async def test_route_does_not_force_multi_from_numbers_or_conjunctions():
    strategy, _ = await route_query("同一表格算手续费和退款，实付83元并且用了16天。",
        FakeLLMProvider(default_response='{"strategy":"single","rationale":"同一规则"}'))
    assert strategy == RetrievalStrategy.SINGLE


@pytest.mark.asyncio
async def test_invalid_route_preserves_existing_offline_fallback():
    strategy, reason = await route_query("一般政策能否适用？", FakeLLMProvider(default_response='{"strategy":9}'))
    assert strategy == RetrievalStrategy.SINGLE and "降级兜底" in reason


@pytest.mark.asyncio
async def test_router_requests_json_with_bounded_output():
    class Recording:
        async def generate(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 256 and kwargs["response_format"] == {"type":"json_object"}
            return LLMResponse(content='{"strategy":"single"}')
    assert (await route_query("一般商品保修期多久？", Recording()))[0] == RetrievalStrategy.SINGLE


@pytest.mark.asyncio
async def test_calculation_trace_reaches_service_response():
    from policy_rag.config import Settings
    from policy_rag.schemas.chat import ChatRequest
    from policy_rag.service import PolicyContainer, PolicyQAService
    class FlowLLM:
        async def generate(self, messages, **kwargs):
            prompt = messages[0]["content"]
            if "检索策略分类器" in prompt:
                content='{"strategy":"single","rationale":"一张费率表"}'
            elif "政策证据评审员" in prompt:
                content='{"sufficient":true}'
            elif "Hallucination Checker" in prompt:
                content='{"supported":true}'
            else:
                content=json.dumps({"action":"answer","answer":"附加费[[calc:freight]]元。",
                                    "calculations":[plan().model_dump()],"citation_chunk_ids":[]})
            return LLMResponse(content=content)
    container=PolicyContainer(settings=Settings(_env_file=None,app_env="testing"),llm=FlowLLM(),chunks=evidence())
    service=PolicyQAService(container)
    response=await service.answer(ChatRequest(message="3.4kg包裹按给定附加费表应付多少？"))
    assert response.answer=="附加费19.00元。"
    assert response.calculation_trace[0]["rendered"]=="19.00"
    assert response.citations[0].chunk_id=="c1"
