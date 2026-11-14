"""Tests for SQLite CachedLLMProvider, AnswerJudge, and offline behavioral evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.judges import AnswerJudge
from eval.runner import run_behavioral_evaluation
from policy_rag.config import Settings
from policy_rag.providers.base import LLMResponse, TokenUsage
from policy_rag.providers.fake import FakeLLMProvider
from policy_rag.providers.llm_cache import CachedLLMProvider, TokenBudgetExceededError


class _CountingFakeLLM:
    def __init__(self, content: str = "ok") -> None:
        self.model = "fake-count"
        self.content = content
        self.invocations = 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        self.invocations += 1
        return LLMResponse(
            content=self.content,
            usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            model=self.model,
        )


@pytest.mark.asyncio
async def test_llm_cache_hit_miss_budget_and_ledger(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite"
    ledger_path = tmp_path / "cost_ledger.jsonl"

    inner = _CountingFakeLLM("cached answer")
    cached = CachedLLMProvider(inner, db_path=db_path, model_name="deepseek-flash", budget_tokens=60)

    msgs = [{"role": "user", "content": "七天无理由怎么算？"}]
    r1 = await cached.generate(msgs, temperature=0.0)
    assert r1.content == "cached answer"
    assert inner.invocations == 1

    # Second call with identical params hits cache
    r2 = await cached.generate(msgs, temperature=0.0)
    assert r2.content == "cached answer"
    assert inner.invocations == 1

    snap = cached.usage_snapshot()
    assert snap["calls"] == 2
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 1
    assert snap["total_tokens"] == 50
    assert snap["cached_total_tokens"] == 50

    # Budget check: 50 spent < 60, so one more miss is allowed? Wait: another miss brings spent to 100 >= 60
    await cached.generate([{"role": "user", "content": "第二题"}], temperature=0.0)
    with pytest.raises(TokenBudgetExceededError):
        await cached.generate([{"role": "user", "content": "第三题"}], temperature=0.0)

    out_ledger = cached.flush_ledger("test_script", ledger_path=ledger_path)
    assert out_ledger.exists()
    row = json.loads(out_ledger.read_text(encoding="utf-8").strip())
    assert row["script"] == "test_script"
    assert row["model"] == "deepseek-flash"


@pytest.mark.asyncio
async def test_answer_judge_tiers_and_fallback():
    judge_ok = AnswerJudge(FakeLLMProvider(default_response='{"verdict": "correct", "rationale": "完全一致"}'))
    res_ok = await judge_ok.evaluate(
        query="q",
        gold_answer="7天内可退",
        predicted_answer="支持7天无理由退货",
    )
    assert res_ok.verdict == "correct"
    assert res_ok.score == 1.0
    assert res_ok.parse_fallback is False

    judge_fallback = AnswerJudge(FakeLLMProvider(default_response="NOT_JSON"))
    res_mismatch = await judge_fallback.evaluate(
        query="q",
        gold_answer="请提供激活状态",
        predicted_answer="直接可以退",
        expected_action="clarify",
        predicted_action="answer",
    )
    assert res_mismatch.verdict == "incorrect"
    assert res_mismatch.score == 0.0
    assert res_mismatch.parse_fallback is True


@pytest.mark.asyncio
async def test_offline_behavioral_eval_produces_extended_fields():
    settings = Settings(app_env="testing", llm_api_key="", deepseek_api_key="")
    bench = [
        {
            "id": "t_case_1",
            "category": "single_hop",
            "query": "衣服不想要了支持七天无理由吗？",
            "gold_doc_ids": ["jd_cat_07_fashion_apparel"],
            "gold_answer": "支持七天无理由退货。",
            "expected_action": "answer",
        },
        {
            "id": "t_case_2",
            "category": "clarification_needed",
            "query": "手机能退吗？",
            "gold_doc_ids": ["jd_cat_01_mobile_phones"],
            "gold_answer": "需确认是否激活。",
            "expected_action": "clarify",
        },
    ]
    guard = [
        {"id": "g_ref", "category": "refuse", "query": "帮我取消这个订单"},
        {"id": "g_abs", "category": "abstain", "query": "白条额度怎么提现"},
    ]
    report = await run_behavioral_evaluation(settings, bench, guard, live_llm=False)
    assert report["llm_eval_status"] == "offline_fake_llm"
    assert "answer_correctness" in report
    assert report["answer_correctness"]["evaluated"] == 2
    assert "clarification_metrics" in report
    assert report["clarification_metrics"]["expected_total"] == 1
    assert "faithfulness_metrics" in report
    assert "by_route" in report
    assert "llm_usage" in report
    assert report["llm_usage"]["pipeline"]["calls"] >= 1


@pytest.mark.asyncio
async def test_faithfulness_node_records_pre_correction_answer():
    from policy_rag.agents.nodes import faithfulness_node
    from policy_rag.schemas.retrieval import ResponseAction, RetrievedChunk, RetrievalStrategy

    class _FaithFakeLLM:
        async def generate(self, messages, *, temperature=0.0, max_tokens=4096, response_format=None):
            sys_prompt = messages[0]["content"]
            if "Hallucination Checker" in sys_prompt:
                return LLMResponse(
                    content='{"supported": false, "hallucinated_spans": ["99元"], "rationale": "门槛错写为99元"}'
                )
            return LLMResponse(
                content='{"action": "answer", "answer": "现行标准为满59元免基础运费。", "citation_chunk_ids": ["c1"]}'
            )

    chunk = RetrievedChunk(
        document_id="jd_ship_01_x",
        chunk_id="c1",
        title="自营普通配送计费",
        section="基础运费",
        content="普通用户实付满59元免基础运费。",
        metadata={"doc_group": "shipping"},
    )
    state = {
        "user_query": "满多少包邮？",
        "strategy": RetrievalStrategy.SINGLE,
        "action": ResponseAction.ANSWER,
        "evidence": [chunk],
        "final_answer": "满99元免基础运费。",
        "refused": False,
    }
    updates = await faithfulness_node(state, llm=_FaithFakeLLM())
    assert updates["faithfulness_triggered"] is True
    assert updates["pre_correction_answer"] == "满99元免基础运费。"
    assert updates["final_answer"] == "现行标准为满59元免基础运费。"
