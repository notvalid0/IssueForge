from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class GenerationContext:
    """插件生成一轮 kernel 所需的全部上下文。

    项目本体只负责采集这些信息；具体如何写 prompt、调用哪个模型、
    是否使用 reflexion/搜索/多候选，由插件自行决定。
    """

    phase: str
    round_index: int
    task_name: str
    task_source: str
    gpu: str
    current_code: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    llm: Any = None
    backend: Any = None
    run_dir: Path = Path(".")
    retrieved: list[Any] = field(default_factory=list)

    @property
    def error(self) -> str:
        return str(self.metrics.get("message") or self.metrics.get("error") or "")


@runtime_checkable
class KernelGenerator(Protocol):
    """Kernel 生成插件协议。

    实现该协议即可通过 entry point 或 ``module:Class`` 接入，
    不需要修改 IssueForge 本体代码。
    """

    name: str

    def diagnose(self, ctx: GenerationContext) -> dict[str, Any]:
        ...

    def generate(self, ctx: GenerationContext) -> str:
        ...


class BaseGenerator:
    """插件基类：提供默认诊断，方便外部实现只关心 generate。"""

    name = "base"

    def diagnose(self, ctx: GenerationContext) -> dict[str, Any]:
        roofline = ctx.metrics.get("roofline") or {}
        return {
            "bottleneck": roofline.get("bottleneck", "unknown"),
            "root_cause": "；".join(ctx.metrics.get("bottleneck_tags") or []) or "NCU 指标不足",
            "recommended_fix": "先做单点改动并重新测量。",
            "evidence": [],
            "avoid": [],
            "try": [],
        }

    def generate(self, ctx: GenerationContext) -> str:  # pragma: no cover - interface
        raise NotImplementedError
