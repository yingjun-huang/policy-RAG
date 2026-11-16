"""Three-tier LLM answer correctness judge against reference gold answers."""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel

from policy_rag.agents.loop import _clean_json_str
from policy_rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """你是一个严谨的电商售后政策评测裁判。请对照“标准参考答案（gold_answer）”和“预期动作（expected_action）”，对“系统回答（predicted_answer）”进行三档判分。

判分标准：
- correct (1.0): 结论方向、核心数值（金额/天数/比例/阈值）、政策版本归属与前置条件完全正确，或预期动作一致且准确回答。
- partial (0.5): 大方向正确，但遗漏了关键前提条件、补充边界或部分次要数值表述不够准确（未出现明显原则性错误）。
- incorrect (0.0): 核心结论错误、数值算错、混淆新旧版本、无依据编造，或该拒答/澄清/留白时给出了错误确定性回答。

只输出标准 JSON：
{
  "verdict": "correct|partial|incorrect",
  "rationale": "简短判定依据（不超过80字）"
}
"""

JUDGE_USER = """用户问题：{query}
预期动作：{expected_action}
系统实际动作：{predicted_action}
标准参考答案：{gold_answer}
系统实际回答：{predicted_answer}

请输出裁判 JSON："""

_VERDICT_SCORE: dict[str, float] = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
}


class AnswerJudgment(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = "incorrect"
    score: float = 0.0
    rationale: str = ""
    parse_fallback: bool = False


class AnswerJudge:
    """Evaluates predicted answers against gold answers with 3-tier grading."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def evaluate(
        self,
        *,
        query: str,
        gold_answer: str,
        predicted_answer: str,
        expected_action: str = "answer",
        predicted_action: str = "answer",
    ) -> AnswerJudgment:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    query=query,
                    expected_action=expected_action,
                    predicted_action=predicted_action,
                    gold_answer=gold_answer,
                    predicted_answer=predicted_answer,
                ),
            },
        ]
        try:
            resp = await self.llm.generate(
                messages,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            data = json.loads(_clean_json_str(resp.content))
            verdict_raw = str(data.get("verdict") or "").strip().lower()
            if verdict_raw in _VERDICT_SCORE:
                return AnswerJudgment(
                    verdict=verdict_raw,  # type: ignore[arg-type]
                    score=_VERDICT_SCORE[verdict_raw],
                    rationale=str(data.get("rationale") or "").strip(),
                    parse_fallback=False,
                )
        except Exception as exc:
            logger.warning("AnswerJudge failed to parse JSON response: %s", exc)

        # Deterministic fallback when mock/non-JSON output is returned in offline tests
        if expected_action != predicted_action and expected_action in {"refuse", "clarify", "abstain"}:
            return AnswerJudgment(
                verdict="incorrect",
                score=0.0,
                rationale=f"动作不匹配: expected={expected_action}, got={predicted_action}",
                parse_fallback=True,
            )
        if predicted_answer.strip() and gold_answer.strip() and gold_answer.strip() == predicted_answer.strip():
            return AnswerJudgment(verdict="correct", score=1.0, rationale="精确一致", parse_fallback=True)
        return AnswerJudgment(
            verdict="partial" if predicted_answer.strip() else "incorrect",
            score=0.5 if predicted_answer.strip() else 0.0,
            rationale="非JSON响应兜底判定",
            parse_fallback=True,
        )
