from __future__ import annotations

import csv
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger("issueforge.ncu")

METRICS = [
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    # roofline SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
]

METRIC_COLUMNS = [item.strip() for item in METRICS]


def ncu_available(ncu_bin: str = "ncu") -> bool:
    return shutil.which(ncu_bin) is not None or Path(ncu_bin).exists()


def ncu_version(ncu_bin: str = "ncu") -> str | None:
    if not ncu_available(ncu_bin):
        return None
    try:
        result = subprocess.run([ncu_bin, "--version"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if line.startswith("Version "):
                return line.split()[1]
    except Exception:  # noqa: BLE001
        return None
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if not text or text in {"-", "N/A", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_ncu_csv(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw:
                continue
            name = raw.get("Kernel Name") or raw.get("Function Name") or ""
            if not name or name.lower() in {"kernel name", "function name"}:
                continue
            row: dict = {"Kernel Name": name}
            for column in METRIC_COLUMNS:
                if column in raw:
                    value = _to_float(raw[column])
                    if value is not None:
                        row[column] = value
            if len(row) > 1:
                rows.append(row)
    return rows


def select_kernel(rows: list[dict], kernel_names: list[str] | None = None) -> dict:
    if not rows:
        return {}
    if kernel_names:
        for name in kernel_names:
            for row in rows:
                if name and name in str(row.get("Kernel Name", "")):
                    return row
    # 跳过 PyTorch 内部 kernel
    for row in rows:
        if not str(row.get("Kernel Name", "")).startswith(("at::", "void at::")):
            return row
    return rows[-1]


def metrics_to_prompt(metrics: dict, max_metrics: int = 20) -> str:
    if not metrics:
        return "（NCU 没有返回可用指标）"
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    for key, value in list(metrics.items())[:max_metrics]:
        if key == "Kernel Name":
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _metric(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Roofline:
    compute_sol_pct: float
    memory_sol_pct: float
    efficiency_pct: float
    headroom_pct: float
    bottleneck: str
    uses_tensor_cores: bool
    at_roofline: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def roofline(metrics: dict, threshold: float = 95.0) -> Roofline:
    compute = _metric(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    memory = _metric(metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    warnings: list[str] = []
    if not compute and not memory:
        warnings.append("缺少 SOL 指标")
    efficiency = max(compute, memory)
    if efficiency < 35.0:
        bottleneck = "underutilized"
    elif memory >= compute:
        bottleneck = "memory"
    else:
        bottleneck = "compute"
    return Roofline(
        compute_sol_pct=round(compute, 2),
        memory_sol_pct=round(memory, 2),
        efficiency_pct=round(efficiency, 2),
        headroom_pct=round(max(0.0, 100.0 - efficiency), 2),
        bottleneck=bottleneck,
        uses_tensor_cores=_metric(metrics, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active") > 5,
        at_roofline=efficiency >= threshold,
        warnings=warnings,
    )


def derive_bottlenecks(metrics: dict) -> list[str]:
    tags: list[str] = []
    if not metrics:
        return tags
    warps = _metric(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active")
    regs = _metric(metrics, "launch__registers_per_thread")
    if warps and warps < 45:
        tags.append("low-occupancy")
    if regs >= 64:
        tags.append("register-pressure")
    if _metric(metrics, "dram__throughput.avg.pct_of_peak_sustained_elapsed") >= 70:
        tags.append("memory-bound")
    if _metric(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed") >= 70:
        tags.append("compute-bound")
    if _metric(metrics, "l1tex__t_sector_hit_rate.pct") and _metric(metrics, "l1tex__t_sector_hit_rate.pct") < 60:
        tags.append("low-l1-hit-rate")
    if _metric(metrics, "lts__t_sector_hit_rate.pct") and _metric(metrics, "lts__t_sector_hit_rate.pct") < 60:
        tags.append("low-l2-hit-rate")
    if _metric(metrics, "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct") >= 15:
        tags.append("long-scoreboard-stall")
    if _metric(metrics, "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct") >= 15:
        tags.append("short-scoreboard-stall")
    if _metric(metrics, "smsp__warp_issue_stalled_barrier_per_warp_active.pct") >= 10:
        tags.append("barrier-stall")
    if _metric(metrics, "smsp__sass_average_branch_targets_threads_uniform.pct") and _metric(
        metrics, "smsp__sass_average_branch_targets_threads_uniform.pct"
    ) < 80:
        tags.append("branch-divergence")
    if _metric(metrics, "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active") < 5:
        tags.append("tensor-core-idle")
    return tags


def profile(
    bench_script: Path,
    out_csv: Path,
    *,
    ncu_bin: str = "ncu",
    kernel_names: list[str] | None = None,
    timeout: int = 600,
) -> Path:
    """运行 NCU。调用方负责确保 GPU 与权限可用。"""
    if not ncu_available(ncu_bin):
        raise RuntimeError(f"找不到 NCU：{ncu_bin}")
    cmd = [
        ncu_bin,
        "--csv",
        "--page=raw",
        "--kernel-name-base=demangled",
        "--target-processes=all",
        "--replay-mode=kernel",
        "--profile-from-start=on",
        f"--log-file={out_csv}",
        f"--metrics={','.join(METRIC_COLUMNS)}",
        "--launch-skip=0",
        "--launch-count=20",
    ]
    if kernel_names:
        names = sorted({name for name in kernel_names if name})
        if len(names) == 1:
            cmd.append(f"--kernel-name={names[0]}")
        elif names:
            import re

            pattern = "|".join(re.escape(name) for name in names)
            cmd.append(f"--kernel-name=::regex:^({pattern})(\\(|$)")
    cmd.extend(["python", str(bench_script)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"NCU 失败（{result.returncode}）：{result.stderr[-800:]}")
    return out_csv


def synthetic_metrics(round_index: int, speedup: float, bottleneck: str = "memory") -> dict:
    """无 GPU 时的可复现实验数据，用于跑通 Agent 与 Dashboard。"""
    progress = max(0.0, min(1.0, (speedup - 1.0) / 3.0))
    memory = min(92.0, 42.0 + 30.0 * progress + (round_index % 3) * 2)
    compute = min(90.0, 30.0 + 35.0 * progress + (round_index % 2) * 3)
    if bottleneck == "compute":
        compute, memory = memory, compute
    warps = min(88.0, 35.0 + 28.0 * progress)
    return {
        "Kernel Name": "mock_kernel",
        "sm__cycles_active.avg": 120000.0 - 9000 * round_index,
        "sm__warps_active.avg.pct_of_peak_sustained_active": warps,
        "launch__occupancy_limit_blocks": 8.0,
        "launch__occupancy_limit_registers": 6.0 if round_index % 2 else 8.0,
        "launch__occupancy_limit_shared_mem": 8.0,
        "launch__registers_per_thread": 72.0 - 4 * round_index,
        "sm__inst_executed.sum": 3.2e6,
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active": compute * 0.7,
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": 3.0 + 4 * progress,
        "dram__bytes_read.sum": 8.0e8,
        "dram__bytes_write.sum": 4.0e8,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": memory,
        "dram__bytes.sum.per_second": 1.2e11,
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": memory,
        "l1tex__t_sector_hit_rate.pct": 58.0 + 18.0 * progress,
        "l1tex__throughput.avg.pct_of_peak_sustained_active": 45.0 + 20.0 * progress,
        "lts__t_sector_hit_rate.pct": 52.0 + 22.0 * progress,
        "lts__throughput.avg.pct_of_peak_sustained_active": 48.0 + 20.0 * progress,
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": 22.0 - 6 * progress,
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": 18.0 - 5 * progress,
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": 24.0 - 8 * progress,
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct": 8.0,
        "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct": 4.0,
        "smsp__sass_average_branch_targets_threads_uniform.pct": 88.0,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": compute,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": memory,
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 3.0 + 4 * progress,
    }


def finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0
