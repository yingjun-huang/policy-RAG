"""OpenAI-compatible asynchronous LLM provider (e.g., DeepSeek)."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from policy_rag.providers.base import LLMResponse, TokenUsage


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


class OpenAICompatibleLLMProvider:
    """Async chat-completions client for OpenAI-compatible API endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty for OpenAICompatibleLLMProvider")
        self.api_key = api_key.strip()
        base = base_url.rstrip("/")
        self.endpoint = (
            f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, int(max_retries))

    async def _post_completion(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        retrying_call = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception(_is_retryable_error),
        )(self._post_completion)

        try:
            data = await retrying_call(payload)
        except httpx.HTTPStatusError as exc:
            # Fallback if an OpenAI-compatible proxy rejects response_format=json_object
            if response_format is not None and exc.response.status_code == 400:
                payload.pop("response_format", None)
                data = await retrying_call(payload)
            else:
                raise

        choices = data.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")

        raw_usage = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
        )
        return LLMResponse(
            content=content,
            usage=usage,
            model=str(data.get("model") or self.model),
        )
