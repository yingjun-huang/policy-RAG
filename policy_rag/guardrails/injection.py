"""Prompt injection and jailbreak detection guardrail."""

from __future__ import annotations

import re

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"忽略(以上|之前|上面)(的)?(指令|提示|规则)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"exfiltrat|send\s+all\s+(user\s+)?(data|information)", re.I),
    re.compile(r"(泄露|输出|打印).*(系统提示词|system\s*prompt)", re.I),
)


def detect_prompt_injection(text: str) -> bool:
    """Returns True if the text contains obvious prompt injection or jailbreak attempts."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
