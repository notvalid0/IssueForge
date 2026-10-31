from __future__ import annotations

import json

import httpx
import pytest

from fastapi.testclient import TestClient

from issueforge_core.context import format_context
from issueforge_server.ingest.demo import seed_demo
from issueforge_server.kb.embed import HashEmbedder
from issueforge_server.kb.search import SearchResult, card_document, search_cards, search_with_rerank
from issueforge_server.llm.rerank import RerankClient, RerankError
from issueforge_server.models import KnowledgeCard
from issueforge_server.web.app import create_app


class FakeReranker:
    """测试替身：返回固定的重排顺序。"""

    def __init__(
        self,
        order: list[int] | None = None,
        scores: list[float] | None = None,
        fail: bool = False,
    ) -> None:
        self.order = order
        self.scores = scores
        self.fail = fail
        self.calls: list[dict] = []

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        self.calls.append({"query": query, "n": len(documents), "top_n": top_n})
        if self.fail:
            raise RuntimeError("rerank 服务不可用")
        count = len(documents) if top_n is None else min(top_n, len(documents))
        indices = (self.order or list(range(len(documents))))[:count]
        scores = self.scores or [1.0 / (rank + 1) for rank in range(len(indices))]
        return [(index, scores[rank]) for rank, index in enumerate(indices)]


# ---------------- RerankClient ----------------


def _client(handler, **kwargs) -> RerankClient:
    transport = httpx.MockTransport(handler)
    return RerankClient(
        "https://gw.test/v1/rerank",
        "test-key",
        "jina-reranker-m0",
        transport=transport,
        **kwargs,
    )


def test_rerank_client_parsing_and_payload() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.32},
                ]
            },
        )

    client = _client(handler)
    results = client.rerank("bank conflict", ["doc a", "doc b"], top_n=2)
    assert seen["url"] == "https://gw.test/v1/rerank"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "jina-reranker-m0"
    assert seen["body"]["query"] == "bank conflict"
    assert seen["body"]["documents"] == ["doc a", "doc b"]
    assert seen["body"]["top_n"] == 2
    # 按分数降序
    assert results == [(1, 0.91), (0, 0.32)]


def test_rerank_client_top_n_truncates_and_sorts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
        )

    client = _client(handler)
    results = client.rerank("q", ["a", "b", "c"], top_n=2)
    assert [index for index, _ in results] == [0, 1]


def test_rerank_client_accepts_score_alias() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.77}]})

    client = _client(handler)
    assert client.rerank("q", ["a"]) == [(0, 0.77)]


def test_rerank_client_unconfigured_raises() -> None:
    client = RerankClient("", "", "jina-reranker-m0")
    assert client.configured is False
    with pytest.raises(RerankError):
        client.rerank("q", ["a"])


def test_rerank_client_http_error_raises() -> None:
    client = _client(lambda request: httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(RerankError):
        client.rerank("q", ["a"])


def test_rerank_client_retries_on_500() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    client = _client(handler, max_retries=1)
    assert client.rerank("q", ["a"]) == [(0, 0.5)]
    assert calls["n"] == 2


def test_rerank_client_empty_documents() -> None:
    client = _client(lambda request: httpx.Response(200, json={"results": []}))
    assert client.rerank("q", []) == []


# ---------------- search_with_rerank ----------------


def _ids(results: list[SearchResult]) -> list[int]:
    return [int(r.card["id"]) for r in results]


def test_search_with_rerank_reorders(session, embedder) -> None:
    seed_demo(session, embedder)
    base = search_cards(session, embedder, "bank conflict", limit=5)
    assert len(base) >= 2
    first, second = base[0], base[1]

    # 故意把第二个候选排到最前
    reranker = FakeReranker(order=[1, 0], scores=[0.9, 0.2])
    results = search_with_rerank(session, embedder, "bank conflict", limit=5, reranker=reranker)

    assert _ids(results)[0] == int(second.card["id"])
    assert results[0].rerank_score == 0.9
    assert results[1].rerank_score == 0.2
    # rerank 请求确实携带了候选文档
    assert reranker.calls[0]["n"] >= 2
    assert reranker.calls[0]["top_n"] == 5


def test_search_with_rerank_failure_falls_back(session, embedder) -> None:
    seed_demo(session, embedder)
    base = search_cards(session, embedder, "bank conflict", limit=5)
    results = search_with_rerank(
        session, embedder, "bank conflict", limit=5, reranker=FakeReranker(fail=True)
    )
    # 失败时退回原始排序，且不携带 rerank 分数
    assert _ids(results) == _ids(base)
    assert all(item.rerank_score is None for item in results)


def test_search_without_reranker_unchanged(session, embedder) -> None:
    seed_demo(session, embedder)
    base = search_cards(session, embedder, "bank conflict", limit=5)
    results = search_with_rerank(session, embedder, "bank conflict", limit=5, reranker=None)
    assert _ids(results) == _ids(base)
    assert all(item.rerank_score is None for item in results)


def test_rerank_backfills_when_fewer_results(session, embedder) -> None:
    seed_demo(session, embedder)
    base = search_cards(session, embedder, "bank conflict", limit=5)
    # rerank 只返回一条，其余用原始排序补齐
    reranker = FakeReranker(order=[1], scores=[0.99])
    results = search_with_rerank(session, embedder, "bank conflict", limit=5, reranker=reranker)
    assert _ids(results)[0] == _ids(base)[1]
    assert results[0].rerank_score == 0.99
    assert len(results) == len(base)
    assert results[1].rerank_score is None


def test_card_document_includes_key_fields() -> None:
    doc = card_document(
        {
            "issue_title": "bank conflict in tiled matmul",
            "one_line": "padding 到 33",
            "root_cause": "共享内存列对齐",
            "approach": "padding",
            "tags": ["bank-conflict"],
        }
    )
    for fragment in ("bank conflict in tiled matmul", "padding 到 33", "共享内存列对齐", "bank-conflict"):
        assert fragment in doc


def test_format_context_shows_rerank_score() -> None:
    item = SearchResult(
        card={"id": 1, "repo": "acme/cuda", "issue_number": 7, "one_line": "x", "issue_title": "t"},
        score=0.5,
        rerank_score=0.8766,
    )
    text = format_context([item])
    assert "rerank 0.877" in text
    # 无 rerank 分数时不显示
    item.rerank_score = None
    assert "rerank" not in format_context([item])


# ---------------- API 集成 ----------------


def test_api_search_returns_rerank_score(env, monkeypatch: pytest.MonkeyPatch) -> None:
    import issueforge_server.web.app as app_module

    fake = FakeReranker(order=None, scores=[0.9, 0.4, 0.1])

    class FakeClient:
        """替换 app 模块里的 RerankClient：不做网络请求。"""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def rerank(self, query: str, documents: list[str], top_n: int | None = None):
            return fake.rerank(query, documents, top_n)

        def test(self) -> dict:
            return {"ok": True, "top": []}

        def close(self) -> None:
            pass

    monkeypatch.setattr(app_module, "RerankClient", FakeClient)
    monkeypatch.delenv("BUPT_API_KEY", raising=False)

    app = create_app(env, start_worker=False)
    with TestClient(app) as client:
        client.post("/api/demo/seed")
        saved = client.put("/api/settings", json={"rerank_api_key": "test-rerank-key"})
        assert saved.status_code == 200
        assert saved.json()["has_rerank_api_key"] is True

        response = client.post("/api/kb/search", json={"query": "bank conflict", "limit": 3})
        assert response.status_code == 200
        items = response.json()["items"]
        assert items
        assert all("rerank_score" in item for item in items)
        scores = [item["rerank_score"] for item in items]
        assert scores == sorted(scores, reverse=True)

        # test-rerank 端点
        test_response = client.post("/api/settings/test-rerank")
        assert test_response.status_code == 200
        assert test_response.json()["ok"] is True

        # Agent API 也带 rerank 分数
        token = client.post("/api/settings/agent-token").json()["token"]
        agent_response = client.post(
            "/api/agent/kb/search",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "bank conflict", "limit": 3},
        )
        assert agent_response.status_code == 200
        assert all("rerank_score" in item for item in agent_response.json()["items"])


def test_api_test_rerank_requires_config(env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUPT_API_KEY", raising=False)
    app = create_app(env, start_worker=False)
    with TestClient(app) as client:
        response = client.post("/api/settings/test-rerank")
        assert response.status_code == 400
