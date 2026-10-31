from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from issueforge_server.web.app import create_app


def _agent_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_api_end_to_end(env, example_task: Path) -> None:
    app = create_app(env, start_worker=False)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        seeded = client.post("/api/demo/seed")
        assert seeded.status_code == 200
        assert seeded.json()["created_cards"] == 3

        cards = client.get("/api/kb/cards")
        assert cards.status_code == 200
        assert cards.json()["total"] == 3

        search = client.post("/api/kb/search", json={"query": "bank conflict", "limit": 5})
        assert search.status_code == 200
        assert search.json()["items"]

        created = client.post(
            "/api/groups",
            json={"name": "测试组", "description": "demo", "criteria": "CUDA kernel", "threshold": 0.5},
        )
        assert created.status_code == 200
        group_id = created.json()["id"]
        assert client.get(f"/api/groups/{group_id}").json()["name"] == "测试组"

        patched = client.patch(f"/api/groups/{group_id}", json={"description": "updated"})
        assert patched.status_code == 200
        assert patched.json()["description"] == "updated"

        assert client.delete(f"/api/groups/{group_id}").status_code == 200
        assert client.get(f"/api/groups/{group_id}").status_code == 404

        settings = client.get("/api/settings")
        assert settings.status_code == 200
        assert "embedding_backend" in settings.json()
        saved = client.put("/api/settings", json={"llm_model": "test-model"})
        assert saved.status_code == 200
        assert saved.json()["llm_model"] == "test-model"

        # ---- Agent API：Token 鉴权 + RAG + 运行回传 ----
        unauth = client.get("/api/agent/ping")
        assert unauth.status_code == 503  # 尚未生成 Token

        token_response = client.post("/api/settings/agent-token")
        assert token_response.status_code == 200
        token = token_response.json()["token"]
        assert token

        bad_token = client.get("/api/agent/ping", headers=_agent_headers("wrong"))
        assert bad_token.status_code == 401

        ping = client.get("/api/agent/ping", headers=_agent_headers(token))
        assert ping.status_code == 200
        assert ping.json()["ok"] is True

        agent_search = client.post(
            "/api/agent/kb/search",
            headers=_agent_headers(token),
            json={"query": "bank conflict", "limit": 3},
        )
        assert agent_search.status_code == 200
        assert agent_search.json()["items"]

        register = client.post(
            "/api/agent/runs",
            headers=_agent_headers(token),
            json={
                "client_run_id": "local-001",
                "name": "agent run",
                "task_name": "task_relu",
                "task_path": str(example_task),
                "gpu": "Mock H100",
                "backend": "mock",
                "config": {"max_rounds": 2, "use_rag": True, "generator": "llm"},
            },
        )
        assert register.status_code == 200
        run_id = register.json()["run_id"]

        # 同一 client_run_id 重复注册返回同一个 run
        re_register = client.post(
            "/api/agent/runs",
            headers=_agent_headers(token),
            json={"client_run_id": "local-001", "name": "agent run"},
        )
        assert re_register.json()["run_id"] == run_id

        round_one = client.post(
            f"/api/agent/runs/{run_id}/rounds",
            headers=_agent_headers(token),
            json={
                "round_index": 0,
                "phase": "seed",
                "metrics": {"runnable": True, "score": 1.12},
                "score": 1.12,
                "runnable": True,
                "retrieved": [1],
                "accepted": True,
                "kernel_code": "class ModelNew:\n    pass\n",
            },
        )
        assert round_one.status_code == 200

        final = client.patch(
            f"/api/agent/runs/{run_id}",
            headers=_agent_headers(token),
            json={
                "status": "done",
                "best_score": 1.24,
                "best_code": "class ModelNew:\n    pass\n",
                "summary": {"best_score": 1.24, "rounds": 1},
                "finished": True,
            },
        )
        assert final.status_code == 200
        assert final.json()["status"] == "done"

        # ---- Dashboard 只读展示回传的运行 ----
        runs = client.get("/api/kernel/runs")
        assert runs.status_code == 200
        assert any(item["id"] == run_id for item in runs.json())

        detail = client.get(f"/api/kernel/runs/{run_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["status"] == "done"
        assert body["best_score"] == 1.24
        assert body["best_code"]
        assert len(body["rounds"]) == 1
        assert body["rounds"][0]["phase"] == "seed"
        assert body["summary"]["rounds"] == 1

        overview = client.get("/api/overview")
        assert overview.status_code == 200
        assert overview.json()["cards"] == 3

        process = client.get("/api/process")
        assert process.status_code == 200
        assert "queue" in process.json()

        assert client.delete(f"/api/kernel/runs/{run_id}").status_code == 200
        assert client.get(f"/api/kernel/runs/{run_id}").status_code == 404

        page = client.get("/")
        assert page.status_code == 200
        assert "IssueForge" in page.text


def test_agent_search_rejects_missing_token(env) -> None:
    app = create_app(env, start_worker=False)
    with TestClient(app) as client:
        client.post("/api/demo/seed")
        response = client.post("/api/agent/kb/search", json={"query": "bank conflict"})
        assert response.status_code == 503
