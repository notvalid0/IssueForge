from __future__ import annotations

import json
import logging

from issueforge_agent.code import extract_code
from issueforge_agent.plugins.base import BaseGenerator, GenerationContext
from issueforge_agent.plugins.prompts import (
    build_diagnose_prompt,
    build_optimize_prompt,
    build_repair_prompt,
    build_seed_prompt,
)

log = logging.getLogger("issueforge_agent.generator.llm")


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-6:]:
        lines.append(
            f"- Round {item.get('round')} [{item.get('phase')}] score={item.get('score')} "
            f"bottleneck={item.get('bottleneck', '')} accepted={item.get('accepted')}"
        )
    return "\n".join(lines)


def rule_diagnosis(metrics: dict) -> dict:
    tags = list(metrics.get("bottleneck_tags") or [])
    roofline = metrics.get("roofline") or {}
    mapping = {
        "low-occupancy": "提高 occupancy：调整 block size、减少寄存器/共享内存占用，或使用 __launch_bounds__。",
        "register-pressure": "降低寄存器压力：减少临时变量、拆分 kernel 或设置 launch_bounds。",
        "memory-bound": "提高内存效率：合并访问、向量化 load/store、使用 shared memory 复用数据。",
        "compute-bound": "提高计算效率：减少冗余指令、使用更宽的向量指令或 Tensor Core。",
        "long-scoreboard-stall": "隐藏长延迟：增加并行 warp、预取数据、使用 cp.async/TMA。",
        "short-scoreboard-stall": "减少 shared memory 依赖：检查 bank conflict，调整 padding/swizzle。",
        "low-l1-hit-rate": "改善 L1 复用：调整 tile 大小与访问顺序。",
        "low-l2-hit-rate": "改善 L2 复用：合并相邻 block 的数据访问，考虑 persistent kernel。",
        "barrier-stall": "减少 __syncthreads：拆分阶段或使用 warp-level 原语。",
        "branch-divergence": "减少分支发散：把边界检查移到统一路径或使用 predication。",
        "tensor-core-idle": "评估 Tensor Core：如果 dtype/形状允许，使用 mma/wgmma 或 CUTLASS。",
    }
    fixes = [mapping[tag] for tag in tags if tag in mapping]
    return {
        "bottleneck": roofline.get("bottleneck") or "underutilized",
        "root_cause": "；".join(tags) or "NCU 指标不足以定位单一根因",
        "recommended_fix": "；".join(fixes[:2]) or "先做小步改动并重新测量，避免一次改太多。",
        "evidence": [],
        "avoid": [],
        "try": tags[:4],
    }


class LlmGenerator(BaseGenerator):
    """内置生成策略：LLM（经 Server 代理）+ 规则诊断兜底。

    它只是默认插件，可被任意 entry point 插件替换。
    """

    name = "llm"

    def _available(self, ctx: GenerationContext) -> bool:
        return ctx.llm is not None and bool(getattr(ctx.llm, "configured", True))

    def _messages(self, ctx: GenerationContext) -> list[dict]:
        history_block = _history_block(ctx.history)
        if ctx.phase == "seed":
            return build_seed_prompt(ctx.task_source, ctx.gpu, ctx.task_name, ctx.rag_context)
        if ctx.phase == "repair":
            return build_repair_prompt(
                ctx.task_source,
                ctx.gpu,
                ctx.task_name,
                ctx.current_code,
                ctx.error,
                ctx.rag_context,
            )
        return build_optimize_prompt(
            ctx.task_source,
            ctx.gpu,
            ctx.task_name,
            ctx.current_code,
            ctx.diagnosis,
            _metrics_block(ctx.metrics),
            history_block,
            ctx.rag_context,
        )

    def diagnose(self, ctx: GenerationContext) -> dict:
        if not self._available(ctx):
            return rule_diagnosis(ctx.metrics)
        roofline = ctx.metrics.get("roofline") or {}
        roofline_text = (
            f"compute_sol={roofline.get('compute_sol_pct')}% "
            f"memory_sol={roofline.get('memory_sol_pct')}% "
            f"efficiency={roofline.get('efficiency_pct')}% "
            f"bottleneck={roofline.get('bottleneck')}"
        )
        tags = list(ctx.metrics.get("bottleneck_tags") or [])
        try:
            messages = build_diagnose_prompt(
                ctx.task_source,
                ctx.gpu,
                ctx.task_name,
                ctx.current_code,
                _metrics_block(ctx.metrics),
                roofline_text,
                tags,
                ctx.rag_context,
            )
            _write_json(ctx.run_dir / "prompts" / f"round{ctx.round_index:03d}_{ctx.phase}_diagnose.json", messages)
            return ctx.llm.complete_json(messages, temperature=0.1)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 诊断失败，使用规则诊断：%s", exc)
            return rule_diagnosis(ctx.metrics)

    def generate(self, ctx: GenerationContext) -> str:
        messages = self._messages(ctx)
        _write_json(ctx.run_dir / "prompts" / f"round{ctx.round_index:03d}_{ctx.phase}.json", messages)
        if self._available(ctx):
            try:
                reply = ctx.llm.complete_text(messages, temperature=0.2, max_tokens=16384)
                _write_text(ctx.run_dir / "replies" / f"round{ctx.round_index:03d}_{ctx.phase}.txt", reply)
                code = extract_code(reply)
                if code:
                    return code
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM 生成失败，使用后端兜底：%s", exc)
        if ctx.backend is None:
            raise RuntimeError("没有可用的 LLM，也没有配置后端兜底")
        return ctx.backend.generate(
            json.dumps(messages, ensure_ascii=False),
            ctx.phase,
            ctx.round_index,
            current_code=ctx.current_code,
        )


def _metrics_block(metrics: dict) -> str:
    from issueforge_core import ncu

    return ncu.metrics_to_prompt(metrics)


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


__all__ = ["LlmGenerator", "rule_diagnosis"]
