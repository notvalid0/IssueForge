from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

_RETRYABLE = {429, 500, 502, 503, 504}
_FENCE_RE = re.compile(r"(?s)^```(?:json)?\s*|\s*```$")


class LlmError(RuntimeError):
    pass


class LlmNotConfigured(LlmError):
    pass


def parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = _FENCE_RE.sub("", raw).strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(raw[start : end + 1])
        if isinstance(value, dict):
            return value
    raise LlmError("模型没有返回 JSON 对象")


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class LlmClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = httpx.Client(timeout=timeout, transport=transport)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def close(self) -> None:
        self.http.close()

    def _url(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        if "/v1" in base or base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if not self.configured:
            raise LlmNotConfigured("LLM 未配置：请先在设置里填写 Base URL、模型和 API Key")
        payload: dict = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post(self._url(), headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.8 * (2**attempt))
                continue
            if response.status_code in _RETRYABLE and attempt < self.max_retries:
                time.sleep(0.8 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise LlmError(f"LLM 返回 {response.status_code}：{response.text[:300]}")
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LlmError("LLM 没有返回内容")
            message = choices[0].get("message") or {}
            return ChatResult(content=message.get("content") or "", tool_calls=message.get("tool_calls") or [])
        raise LlmError(f"LLM 请求失败：{last_exc}")

    def complete_text(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens).content

    def complete_json(self, messages: list[dict], temperature: float = 0.2) -> dict:
        result = self.complete(messages, temperature=temperature, json_mode=True)
        return parse_json_object(result.content)

    def test(self) -> str:
        result = self.complete(
            [{"role": "user", "content": "只回复 ok"}],
            temperature=0,
            max_tokens=8,
        )
        return (result.content or "").strip()[:40]
