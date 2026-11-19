"""Domain boundary and business write operation guardrails."""

from __future__ import annotations

import re

_BLOCKED_TOPICS = (
    "炸弹",
    "枪支",
    "毒品",
    "hack",
    "信用卡盗刷",
    "验证码转发",
    "木马",
)

# Specific order IDs (e.g. ORD-100234, or 12+ digits)
_ORDER_ID_PATTERN = re.compile(r"\b(?:ORD[-_]?\d{4,}|\d{12,})\b", re.I)

# Imperative transactional action intents
_TRANSACTIONAL_INTENT_PATTERNS = (
    re.compile(r"(帮我|我要|立刻|马上|请|催一下)?(取消|拦截|催)(这个|我的|该|此|这单|那单|这笔)?(订单|快递|发货)", re.I),
    re.compile(
        r"(?:(帮我|我要|请)?(修改|更改|变动)(收货)?(地址|电话|联系方式)|把(收货)?(地址|电话|联系方式)(改|换)(成|到|为))",
        re.I,
    ),
    re.compile(r"(直接|帮我|马上|立即)(人工)?(退款|打款|转账)", re.I),
)

_POLICY_INQUIRY_TERMS = ("规则", "规定", "政策", "条件", "流程", "说明", "什么情况下", "可以吗", "能吗", "如何", "怎么", "吗")
_IMPERATIVE_DEMAND_CUES = ("帮我", "我要", "立刻", "马上", "不要跟我解释")


_POLICY_EXPLANATION_PREFIX = re.compile(
    r"(?:(?:咨询|了解|解释|说明|介绍|讲解|解读)(?:一下|下)?"
    r"(?:关于)?(?:如何|怎么)?|如何|怎么)\s*$"
)


def check_scope_and_actions(query: str) -> tuple[bool, str]:
    """Evaluates whether the query is safe and within policy Q&A scope.
    Returns (is_refused, refusal_message).
    """
    lowered = query.lower()

    # 1. Illegal or unsafe topics
    if any(topic in lowered for topic in _BLOCKED_TOPICS):
        return True, "抱歉，该问题超出商城服务与政策咨询范围，系统无法回答。"

    # 2. Specific Order ID provided -> always treat as personal order transactional query
    if _ORDER_ID_PATTERN.search(query):
        return (
            True,
            "本系统为商城规则与政策智能问答助手，专注于通用条款解读与咨询，无法查询或操作特定订单。"
            "如需查询或办理具体订单业务，请前往商城 App 首页【我的订单】自助办理，或联系人工客服。"
        )

    # 3. Allow explicit explanations only for the action they describe.
    is_policy_inquiry = any(term in query for term in _POLICY_INQUIRY_TERMS)
    has_imperative_demand = any(cue in query for cue in _IMPERATIVE_DEMAND_CUES)
    for pattern in _TRANSACTIONAL_INTENT_PATTERNS:
        for match in pattern.finditer(query):
            prefix = query[:match.start()]
            is_explanation = (
                _POLICY_EXPLANATION_PREFIX.search(prefix) is not None
                and not any(cue in match.group() for cue in _IMPERATIVE_DEMAND_CUES)
            )
            is_action_request = (
                match.group().startswith(("请", "把", "立即"))
                or re.search(r"(?:然后|接着|请)\s*$", prefix) is not None
            )
            if is_explanation or (is_policy_inquiry and not has_imperative_demand and not is_action_request):
                continue
            return (
                True,
                "本系统为商城规则与政策智能问答助手，专注于条款解读与咨询，不具备直接操作订单的权限。"
                "如需办理【取消订单、修改收货地址、查询具体订单物流或人工退款】，请前往商城 App 首页【我的订单】自助办理，或联系人工客服。"
            )

    return False, ""
