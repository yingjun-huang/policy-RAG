"""Adaptive complexity and strategy router."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal
from pydantic import BaseModel

from policy_rag.agents.prompts import ROUTER_SYSTEM, ROUTER_USER
from policy_rag.guardrails.boundary import check_scope_and_actions
from policy_rag.guardrails.injection import detect_prompt_injection
from policy_rag.providers.base import LLMProvider
from policy_rag.schemas.retrieval import RetrievalStrategy

logger = logging.getLogger(__name__)

_GREETINGS = {"你好", "您好", "在吗", "在不在", "有人吗", "谢谢", "感谢", "再见", "拜拜", "hi", "hello"}
_MULTI_HOP_CUES = (
    "同时",
    "并且",
    "还是",
    "如果",
    "但是已经",
    "虽然",
    "既",
    "又",
    "除了",
    "另外",
    "旧版",
    "新版",
    "新旧",
)


class _RoutePayload(BaseModel):
    strategy: Literal["direct", "single", "multi", "refuse"]
    rationale: str = "LLM routing"



async def route_query(
    query: str,
    llm: LLMProvider,
    *,
    history: list[dict[str, str]] | None = None,
) -> tuple[RetrievalStrategy, str]:
    """Determines the appropriate execution strategy for the user query."""
    text = query.strip()

    # 1. Hard Guardrails: Injection check
    if detect_prompt_injection(text):
        return RetrievalStrategy.REFUSE, "系统检测到非法的指令注入或越权提示，请求已被拦截。"

    # 2. Hard Guardrails: Boundary check & business transactional write actions
    is_refused, refusal_msg = check_scope_and_actions(text)
    if is_refused:
        return RetrievalStrategy.REFUSE, refusal_msg

    # 3. Rule-based fast direct routing for social greetings (only when no multi-turn policy context)
    cleaned_greeting = re.sub(r"[^\w]", "", text.lower())
    if not history and cleaned_greeting in _GREETINGS:
        return RetrievalStrategy.DIRECT, "简单社交问候，无需检索"

    # 4. LLM-based adaptive routing
    prompt_query = text
    if history:
        recent_turns = [
            f"{t.get('role', 'user')}: {t.get('content', '')}"
            for t in history[-4:]
            if isinstance(t, dict) and t.get("content")
        ]
        if recent_turns:
            prompt_query = f"历史对话:\n" + "\n".join(recent_turns) + f"\n当前补充问题: {text}"

    try:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": ROUTER_USER.format(query=prompt_query)},
        ]
        resp = await llm.generate(messages, temperature=0.0, max_tokens=256, response_format={"type": "json_object"})
        # Clean potential markdown fence
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        data = _RoutePayload.model_validate(json.loads(content))
        if data.strategy == "direct":
            # Only the confirmed greeting fast path above may bypass policy evidence.
            return RetrievalStrategy.SINGLE, "未通过纯问候条件，保守转入政策检索"
        return RetrievalStrategy(data.strategy), data.rationale
    except Exception as e:
        logger.warning(f"LLM routing failed or JSON invalid, falling back to rule-based: {e}")

    # 5. Fallback rule routing
    if any(cue in text for cue in _MULTI_HOP_CUES):
        return RetrievalStrategy.MULTI, "规则降级兜底：包含复合条件关键词"
    return RetrievalStrategy.SINGLE, "规则降级兜底：常规单跳检索"
