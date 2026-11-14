"""SQLite-backed LLM response cache with token accounting and cost ledger export."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policy_rag.providers.base import LLMProvider, LLMResponse, TokenUsage


class TokenBudgetExceededError(RuntimeError):
    """Raised when cumulative spent tokens exceed the configured budget."""


class CachedLLMProvider:
    """Wraps an LLMProvider with deterministic SQLite response caching and token tracking."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        db_path: str | Path = ".cache/llm/cache.sqlite",
        model_name: str | None = None,
        budget_tokens: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.inner = inner
        self.db_path = Path(db_path)
        self.model = model_name or str(getattr(inner, "model", "default"))
        self.budget_tokens = budget_tokens
        self.enabled = enabled

        self.calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.cached_prompt_tokens: int = 0
        self.cached_completion_tokens: int = 0
        self.cached_total_tokens: int = 0

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    response_json TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def make_cache_key(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": round(float(temperature), 6),
            "max_tokens": int(max_tokens),
            "response_format": response_format,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_cached(self, cache_key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return LLMResponse.model_validate_json(row[0])

    def _put_cached(
        self,
        cache_key: str,
        *,
        temperature: float,
        response: LLMResponse,
    ) -> None:
        if not self.enabled:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    cache_key, model, temperature, response_json,
                    prompt_tokens, completion_tokens, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    response.model or self.model,
                    float(temperature),
                    response.model_dump_json(),
                    int(response.usage.prompt_tokens),
                    int(response.usage.completion_tokens),
                    int(response.usage.total_tokens),
                    now_iso,
                ),
            )
            conn.commit()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        cache_key = self.make_cache_key(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        cached = self._get_cached(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self.cached_prompt_tokens += cached.usage.prompt_tokens
            self.cached_completion_tokens += cached.usage.completion_tokens
            self.cached_total_tokens += cached.usage.total_tokens
            return cached

        if self.budget_tokens is not None and self.total_tokens >= self.budget_tokens:
            raise TokenBudgetExceededError(
                f"LLM token budget exceeded: spent={self.total_tokens}, budget={self.budget_tokens}"
            )

        self.cache_misses += 1
        resp = await self.inner.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        usage = resp.usage or TokenUsage()
        self.prompt_tokens += int(usage.prompt_tokens)
        self.completion_tokens += int(usage.completion_tokens)
        self.total_tokens += int(usage.total_tokens)

        self._put_cached(cache_key, temperature=temperature, response=resp)
        return resp

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "cache_enabled": self.enabled,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_completion_tokens": self.cached_completion_tokens,
            "cached_total_tokens": self.cached_total_tokens,
        }

    def flush_ledger(
        self,
        script_name: str,
        *,
        ledger_path: str | Path = "eval/reports/cost_ledger.jsonl",
    ) -> Path:
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": script_name,
            **self.usage_snapshot(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path
