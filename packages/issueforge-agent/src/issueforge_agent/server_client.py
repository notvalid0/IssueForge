from __future__ import annotations

import logging
from typing import Any

import httpx

from issueforge_core.schemas import (
    ChatMessage,
    ChatRequest,
    RoundReport,
    RunPatch,
    RunRegister,
    SearchHit,
    SearchRequest,
)
from issueforge_agent.llm.client import parse_json_object

log = logging.getLogger("issueforge_agent.server")


class ServerError(RuntimeError):
    """Server 不可用、鉴权失败或返回错误。"""


class ServerClient:
    """访问 IssueForge Server 的 Agent API（Bearer Token）。

    只负责知识库检索、LLM 代理和运行回传；不执行任何本地 CUDA 逻辑。
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 240.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.http = httpx.Client(timeout=timeout, transport=transport)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def close(self) -> None:
        self.http.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> Any:
        if not self.configured:
            raise ServerError("Server 未配置：请设置 ISSUEFORGE_SERVER_URL 与 ISSUEFORGE_AGENT_TOKEN")
        try:
            response = self.http.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=json,
            )
        except httpx.HTTPError as exc:
            raise ServerError(f"无法连接 Server：{exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:500]
            try:
                body = response.json()
                detail = body.get("detail") or detail
            except ValueError:
                pass
            raise ServerError(f"Server 返回 {response.status_code}：{detail}")
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # ---------------- 基础 ----------------

    def ping(self) -> dict:
        return self._request("GET", "/api/agent/ping")

    # ---------------- RAG ----------------

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        tags: list[str] | None = None,
        hardware: str | None = None,
    ) -> list[SearchHit]:
        body = SearchRequest(query=query, limit=limit, tags=tags or [], hardware=hardware)
        data = self._request("POST", "/api/agent/kb/search", body.model_dump())
        return [SearchHit.model_validate(item) for item in (data or {}).get("items", [])]

    # ---------------- LLM 代理 ----------------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        body = ChatRequest(
            messages=[ChatMessage.model_validate(message) for message in messages],
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        data = self._request("POST", "/api/agent/llm/chat", body.model_dump())
        return str((data or {}).get("content") or "")

    def complete_text(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def complete_json(self, messages: list[dict], temperature: float = 0.2) -> dict:
        return parse_json_object(self.chat(messages, temperature=temperature, json_mode=True))

    def test(self) -> str:
        return self.complete_text([{"role": "user", "content": "只回复 ok"}], temperature=0, max_tokens=8)[:40]

    # ---------------- 运行回传 ----------------

    def register_run(self, body: RunRegister) -> int:
        data = self._request("POST", "/api/agent/runs", body.model_dump())
        run_id = int((data or {}).get("run_id") or 0)
        if run_id <= 0:
            raise ServerError("Server 没有返回 run_id")
        return run_id

    def report_round(self, run_id: int, body: RoundReport) -> None:
        self._request("POST", f"/api/agent/runs/{run_id}/rounds", body.model_dump())

    def patch_run(self, run_id: int, body: RunPatch) -> None:
        self._request("PATCH", f"/api/agent/runs/{run_id}", body.model_dump())
