from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from issueforge_core import ncu
from issueforge_core.context import format_context
from issueforge_core.schemas import RoundReport, RunPatch, RunRegister

from issueforge_agent.code import extract_code
from issueforge_agent.plugins.base import GenerationContext, KernelGenerator
from issueforge_agent.store import LocalRunStore

log = logging.getLogger("issueforge_agent.loop")


class Cancelled(RuntimeError):
    pass


class KernelOptimizer:
    """本地 Kernel 优化调度器。

    它只负责：准备任务 → 本地评测/NCU → 通过 Server 取 RAG → 调用生成插件 →
    回传轮次与最佳 kernel。具体生成方法完全由 ``generator`` 插件决定。
    """

    def __init__(
        self,
        *,
        backend,
        generator: KernelGenerator,
        data_dir: Path,
        task_path: Path,
        name: str = "",
        gpu: str = "Mock H100",
        server=None,
        llm=None,
        max_rounds: int = 8,
        use_rag: bool = True,
        use_llm: bool = True,
        client_run_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.generator = generator
        self.server = server
        self.llm = llm if llm is not None else server
        self.data_dir = Path(data_dir)
        self.task_path = Path(task_path)
        self.task_name = self.task_path.stem
        self.name = name or self.task_name
        self.gpu = gpu
        self.max_rounds = max(1, min(50, max_rounds))
        self.use_rag = use_rag
        self.use_llm = use_llm
        self.cancel_check = cancel_check
        self.client_run_id = client_run_id or (
            f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        self.store = LocalRunStore(self.data_dir / "runs", self.client_run_id, self.name)
        self.server_run_id: int | None = None
        self.best_code = ""
        self.best_score = float("-inf")
        self.history: list[dict[str, Any]] = []
        self.last_metrics: dict[str, Any] = {}
        self.last_code = ""

    # ---------------- 生命周期 ----------------

    def _check_cancel(self) -> None:
        if self.cancel_check and self.cancel_check():
            raise Cancelled("任务已取消")

    def _register(self) -> None:
        if self.server is None:
            return
        try:
            self.server_run_id = self.server.register_run(
                RunRegister(
                    client_run_id=self.client_run_id,
                    name=self.name,
                    task_name=self.task_name,
                    task_path=str(self.task_path),
                    gpu=self.gpu,
                    backend=getattr(self.backend, "name", "unknown"),
                    config={
                        "max_rounds": self.max_rounds,
                        "use_rag": self.use_rag,
                        "use_llm": self.use_llm,
                        "generator": getattr(self.generator, "name", "unknown"),
                    },
                    environment={"client_run_id": self.client_run_id, "local_dir": str(self.store.dir)},
                )
            )
            self.store.update_manifest(server_run_id=self.server_run_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("向 Server 注册运行失败，继续本地运行：%s", exc)

    def _patch(self, body: RunPatch) -> None:
        if self.server is None or self.server_run_id is None:
            return
        try:
            self.server.patch_run(self.server_run_id, body)
        except Exception as exc:  # noqa: BLE001
            log.warning("回传运行状态失败：%s", exc)

    # ---------------- RAG / 回传 ----------------

    def _retrieve(self, query: str, phase: str, round_index: int):
        if not self.use_rag:
            return "（RAG 已关闭）", []
        if self.server is None:
            return "（未连接 Server，无法检索知识库）", []
        hardware = self.gpu if self.gpu not in {"", "Unknown GPU"} else None
        try:
            hits = self.server.search(query, limit=6, hardware=hardware)
        except Exception as exc:  # noqa: BLE001
            log.warning("检索失败：%s", exc)
            hits = []
        return format_context(hits), hits

    def _report_round(
        self,
        round_index: int,
        phase: str,
        code: str,
        metrics: dict,
        retrieved: list,
        accepted: bool,
        error: str = "",
    ) -> None:
        self.store.write_metrics(round_index, phase, metrics)
        if self.server is None or self.server_run_id is None:
            return
        score = metrics.get("score") if metrics.get("runnable") else None
        try:
            self.server.report_round(
                self.server_run_id,
                RoundReport(
                    round_index=round_index,
                    phase=phase,
                    metrics=metrics,
                    score=float(score) if score is not None else None,
                    runnable=bool(metrics.get("runnable")),
                    error=error or str(metrics.get("message") or ""),
                    retrieved=[int(hit.card.get("id")) for hit in retrieved if hit.card.get("id")],
                    accepted=accepted,
                    kernel_code=code,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("回传第 %s 轮失败：%s", round_index, exc)

    # ---------------- 单轮 ----------------

    def _profile(self, code: str, round_index: int) -> dict:
        try:
            return self.backend.profile(
                code,
                self.task_path,
                round_index,
                speedup=max(1.0, self.best_score if self.best_score > 0 else 1.0),
                work_dir=self.store.dir / "ncu",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("NCU 采集失败，使用空指标：%s", exc)
            return {}

    def _context(self, phase: str, round_index: int, task_src: str, rag_context: str) -> GenerationContext:
        return GenerationContext(
            phase=phase,
            round_index=round_index,
            task_name=self.task_name,
            task_source=task_src,
            gpu=self.gpu,
            current_code=self.last_code,
            metrics=self.last_metrics,
            rag_context=rag_context,
            history=self.history,
            llm=self.llm if self.use_llm else None,
            backend=self.backend,
            run_dir=self.store.dir,
        )

    def _prepare_round(self, round_index: int, task_src: str) -> GenerationContext:
        if round_index == 0:
            phase = "seed"
            query = f"{self.task_name} {self.gpu} CUDA kernel implementation common pitfalls"
            rag_context, retrieved = self._retrieve(query, phase, round_index)
        elif not self.last_metrics.get("runnable", False):
            phase = "repair"
            error = str(self.last_metrics.get("message") or self.last_metrics.get("error") or "未知错误")
            query = f"{error[:500]} CUDA compile runtime error fix"
            rag_context, retrieved = self._retrieve(query, phase, round_index)
        else:
            phase = "optimize"
            profile = self._profile(self.last_code, round_index)
            if profile:
                self.last_metrics = {**self.last_metrics, **profile}
            roof = ncu.roofline(self.last_metrics)
            tags = ncu.derive_bottlenecks(self.last_metrics)
            self.last_metrics["roofline"] = roof.to_dict()
            self.last_metrics["bottleneck_tags"] = tags
            query = (
                f"{self.task_name} {self.gpu} {' '.join(tags)} "
                f"{roof.bottleneck} CUDA kernel optimization"
            )
            rag_context, retrieved = self._retrieve(query, phase, round_index)

        ctx = self._context(phase, round_index, task_src, rag_context)
        ctx.retrieved = retrieved
        if phase == "optimize":
            try:
                ctx.diagnosis = self.generator.diagnose(ctx)
            except Exception as exc:  # noqa: BLE001
                log.warning("生成插件诊断失败，使用空诊断：%s", exc)
                ctx.diagnosis = {}
        return ctx

    # ---------------- 主循环 ----------------

    def run(self) -> dict:
        self._register()
        self.store.write_manifest(
            {
                "client_run_id": self.client_run_id,
                "server_run_id": self.server_run_id,
                "name": self.name,
                "task_name": self.task_name,
                "task_path": str(self.task_path),
                "gpu": self.gpu,
                "backend": getattr(self.backend, "name", "unknown"),
                "generator": getattr(self.generator, "name", "unknown"),
                "status": "running",
                "max_rounds": self.max_rounds,
                "use_rag": self.use_rag,
                "use_llm": self.use_llm,
                "started_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
        try:
            prepared = self.backend.prepare(self.task_path)
            task_src = prepared.get("reference", "")
            no_improve = 0
            for round_index in range(self.max_rounds):
                self._check_cancel()
                ctx = self._prepare_round(round_index, task_src)
                retrieved = getattr(ctx, "retrieved", [])

                code = self.generator.generate(ctx)
                metrics = self.backend.evaluate(
                    code,
                    self.task_path,
                    round_index,
                    work_dir=self.store.dir / "eval",
                )
                runnable = bool(metrics.get("runnable"))
                score = float(metrics.get("score") or 0) if runnable else 0.0
                accepted = runnable and score > self.best_score
                if accepted:
                    self.best_score = score
                    self.best_code = code
                    no_improve = 0
                elif runnable:
                    no_improve += 1

                self.last_code = code
                self.last_metrics = metrics
                self.store.write_code(round_index, ctx.phase, code)
                # 回传时附带本轮诊断用的 roofline / 瓶颈标签（来自 NCU 剖析）
                report_metrics = dict(metrics)
                if ctx.metrics.get("roofline"):
                    report_metrics["roofline"] = ctx.metrics["roofline"]
                if ctx.metrics.get("bottleneck_tags"):
                    report_metrics["bottleneck_tags"] = ctx.metrics["bottleneck_tags"]
                self._report_round(round_index, ctx.phase, code, report_metrics, retrieved, accepted)
                self.history.append(
                    {
                        "round": round_index,
                        "phase": ctx.phase,
                        "score": round(score, 4) if runnable else None,
                        "bottleneck": (ctx.metrics.get("roofline") or {}).get("bottleneck")
                        or ("error" if not runnable else "—"),
                        "accepted": accepted,
                    }
                )
                self.store.update_manifest(
                    best_score=self.best_score if self.best_score > float("-inf") else None,
                    history=self.history,
                )
                if no_improve >= 3:
                    log.info("连续 3 轮无提升，提前停止")
                    break

            if self.best_code:
                best_path = self.store.write_best(self.best_code)
                log.info("最佳 kernel 已保存：%s", best_path)
            summary = {
                "best_score": self.best_score if self.best_score > float("-inf") else 0.0,
                "rounds": len(self.history),
                "backend": getattr(self.backend, "name", "unknown"),
                "generator": getattr(self.generator, "name", "unknown"),
                "gpu": self.gpu,
                "rag": self.use_rag,
                "history": self.history,
                "local_dir": str(self.store.dir),
            }
            self.store.update_manifest(status="done", finished_at=datetime.utcnow().isoformat(timespec="seconds"), summary=summary)
            self._patch(
                RunPatch(
                    status="done",
                    best_score=summary["best_score"],
                    best_code=self.best_code,
                    summary=summary,
                    finished=True,
                )
            )
            return summary
        except Cancelled:
            self.store.update_manifest(status="cancelled", finished_at=datetime.utcnow().isoformat(timespec="seconds"))
            self._patch(RunPatch(status="cancelled", finished=True))
            return {"cancelled": True}
        except Exception as exc:  # noqa: BLE001
            log.exception("Kernel run 失败")
            self.store.update_manifest(status="failed", error=str(exc)[:4000])
            self._patch(RunPatch(status="failed", error=str(exc)[:4000], finished=True))
            return {"error": str(exc)}


__all__ = ["KernelOptimizer", "Cancelled", "extract_code"]
