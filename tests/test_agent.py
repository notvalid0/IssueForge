from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from issueforge_core.schemas import SearchHit
from issueforge_agent.kernel.mock import MockBackend
from issueforge_agent.loop import KernelOptimizer, extract_code
from issueforge_agent.plugins import GeneratorError, create_generator, list_generators
from issueforge_agent.plugins.llm import LlmGenerator


def test_extract_code_prefers_model_new() -> None:
    text = "```python\nprint('noise')\n```\n```python\nclass ModelNew:\n    pass\n```"
    assert extract_code(text).startswith("class ModelNew")


class EchoGenerator:
    """极简自定义插件：只实现 generate，用于验证插件协议。"""

    name = "echo"

    def diagnose(self, ctx: Any) -> dict:
        return {"bottleneck": "memory"}

    def generate(self, ctx: Any) -> str:
        return f"class ModelNew:\n    pass  # echo round {ctx.round_index}\n"


class FakeServer:
    """内存版 Server API：记录 Agent 回传，返回固定检索结果。"""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.searches: list[str] = []
        self.registered: list[Any] = []
        self.rounds: list[Any] = []
        self.patches: list[Any] = []
        self._run_id = 0

    def search(self, query: str, *, limit: int = 6, tags=None, hardware=None) -> list[SearchHit]:
        self.searches.append(query)
        return self.hits

    def register_run(self, body: Any) -> int:
        self.registered.append(body)
        self._run_id += 1
        return self._run_id

    def report_round(self, run_id: int, body: Any) -> None:
        self.rounds.append(body)

    def patch_run(self, run_id: int, body: Any) -> None:
        self.patches.append(body)


def test_mock_optimizer_runs_offline(tmp_path: Path, example_task: Path) -> None:
    optimizer = KernelOptimizer(
        backend=MockBackend(gpu="Mock H100"),
        generator=create_generator("llm"),
        data_dir=tmp_path,
        task_path=example_task,
        gpu="Mock H100",
        server=None,
        llm=None,
        max_rounds=3,
        use_rag=False,
        use_llm=False,
        client_run_id="offline-1",
    )
    summary = optimizer.run()

    assert summary["rounds"] == 3
    assert summary["best_score"] > 1.0
    assert any(item["accepted"] for item in summary["history"])

    run_dir = Path(summary["local_dir"])
    assert "offline-1" in run_dir.name
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert (run_dir / "best_kernel.py").exists()
    assert len(list((run_dir / "code").glob("round*.py"))) == 3
    assert len(list((run_dir / "metrics").glob("round*.json"))) == 3


def test_mock_optimizer_reports_to_server(tmp_path: Path, example_task: Path) -> None:
    hits = [
        SearchHit(
            card={
                "id": 7,
                "repo": "issueforge/demo",
                "issue_number": 1,
                "issue_title": "bank conflict in tiled matmul",
                "one_line": "把 shared tile 的 leading dimension padding 到 33",
                "approach": "padding + 向量 load",
                "tags": ["bank-conflict"],
            },
            score=0.9,
        )
    ]
    server = FakeServer(hits=hits)
    optimizer = KernelOptimizer(
        backend=MockBackend(gpu="Mock H100"),
        generator=create_generator("llm"),
        data_dir=tmp_path,
        task_path=example_task,
        gpu="Mock H100",
        server=server,
        llm=None,
        max_rounds=3,
        use_rag=True,
        use_llm=False,
        client_run_id="local-042",
    )
    summary = optimizer.run()

    assert summary["rounds"] == 3
    assert len(server.registered) == 1
    assert server.registered[0].client_run_id == "local-042"
    assert server.registered[0].gpu == "Mock H100"
    assert len(server.rounds) == 3
    assert any(round_.accepted for round_ in server.rounds)
    assert any(round_.retrieved == [7] for round_ in server.rounds)
    assert server.searches, "RAG 开启时应向 Server 发起检索"
    assert server.patches[-1].status == "done"
    assert server.patches[-1].best_score == summary["best_score"]


def test_custom_plugin_drives_loop(tmp_path: Path, example_task: Path) -> None:
    optimizer = KernelOptimizer(
        backend=MockBackend(gpu="Mock H100"),
        generator=EchoGenerator(),
        data_dir=tmp_path,
        task_path=example_task,
        gpu="Mock H100",
        server=None,
        llm=None,
        max_rounds=2,
        use_rag=False,
        use_llm=False,
        client_run_id="plugin-1",
    )
    summary = optimizer.run()

    assert summary["generator"] == "echo"
    best = (Path(summary["local_dir"]) / "best_kernel.py").read_text(encoding="utf-8")
    assert "echo" in best


def test_builtin_and_spec_generators() -> None:
    assert isinstance(create_generator("llm"), LlmGenerator)
    by_spec = create_generator(plugin_spec="issueforge_agent.plugins.llm:LlmGenerator")
    assert isinstance(by_spec, LlmGenerator)

    names = [item["name"] for item in list_generators()]
    assert "llm" in names

    try:
        create_generator("不存在的插件")
    except GeneratorError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未知插件名应抛出 GeneratorError")


def test_env_plugin_loader(monkeypatch) -> None:
    monkeypatch.setenv("ISSUEFORGE_GENERATOR_PLUGINS", "test_agent:EchoGenerator")
    generator = create_generator("EchoGenerator")
    assert generator.name == "echo"
    names = [item["name"] for item in list_generators()]
    assert "EchoGenerator" in names
