from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger("issueforge_server.rerank")

# 默认指向 BUPT LLM 网关的 rerank 端点（Cohere/Jina 兼容协议）
DEFAULT_RERANK_ENDPOINT = "https://myai.bupt.edu.cn/llm-gw/v1/rerank"
DEFAULT_RERANK_MODEL = "jina-reranker-m0"

_RETRYABLE = {429, 500, 502, 503, 504}


class RerankError(RuntimeError):
    pass


class RerankClient:
    """Rerank 客户端：对 RAG 候选集按 query 做相关性重排序。

    协议（Cohere/Jina 兼容）：
        POST {endpoint}
        {"model": ..., "query": ..., "documents": [...], "top_n": N}
        -> {"results": [{"index": 0, "relevance_score": 0.98}, ...]}
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_RERANK_ENDPOINT,
        api_key: str = "",
        model: str = DEFAULT_RERANK_MODEL,
        timeout: float = 30.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or DEFAULT_RERANK_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = httpx.Client(timeout=timeout, transport=transport)

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.model)

    def close(self) -> None:
        self.http.close()

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """返回 [(原始下标, 相关性分数)]，按分数降序，最多 top_n 条。"""
        if not self.configured:
            raise RerankError("Rerank 未配置：请填写 Rerank API Key")
        if not documents:
            return []
        top = max(1, min(int(top_n or len(documents)), len(documents)))
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post(self.endpoint, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
                continue
            if response.status_code in _RETRYABLE and attempt < self.max_retries:
                time.sleep(0.5 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise RerankError(f"Rerank 返回 {response.status_code}：{response.text[:300]}")
            try:
                data = response.json()
            except ValueError as exc:
                raise RerankError("Rerank 响应不是 JSON") from exc
            results = data.get("results") or []
            scored: list[tuple[int, float]] = []
            for item in results:
                try:
                    index = int(item.get("index"))
                    score = float(item.get("relevance_score", item.get("score", 0.0)))
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(documents):
                    scored.append((index, score))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            return scored[:top]
        raise RerankError(f"Rerank 请求失败：{last_exc}")

    def test(self) -> dict[str, Any]:
        """用一个最小样例验证连通性。"""
        results = self.rerank(
            "shared memory bank conflict",
            ["Padding the shared tile to 33 removes bank conflicts.",
             "The university is located in Beijing."],
            top_n=1,
        )
        return {"ok": True, "top": [{"index": i, "score": round(s, 4)} for i, s in results]}
