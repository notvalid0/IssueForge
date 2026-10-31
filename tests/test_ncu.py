from __future__ import annotations

from pathlib import Path

from issueforge_core import ncu


def test_synthetic_metrics_are_consistent() -> None:
    metrics = ncu.synthetic_metrics(0, 1.0)
    assert metrics["Kernel Name"] == "mock_kernel"
    roof = ncu.roofline(metrics)
    assert roof.bottleneck == "memory"
    assert 0 <= roof.efficiency_pct <= 100
    tags = ncu.derive_bottlenecks(metrics)
    assert "low-occupancy" in tags
    assert "register-pressure" in tags
    assert "tensor-core-idle" in tags


def test_roofline_marks_underutilized() -> None:
    metrics = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 25.0,
    }
    roof = ncu.roofline(metrics)
    assert roof.bottleneck == "underutilized"
    assert roof.at_roofline is False


def test_parse_ncu_csv(tmp_path: Path) -> None:
    path = tmp_path / "ncu.csv"
    path.write_text(
        "Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed\n"
        "mock_kernel,72.5,81.0\n"
        "at::native::copy,10.0,20.0\n",
        encoding="utf-8",
    )
    rows = ncu.parse_ncu_csv(path)
    assert len(rows) == 2
    assert rows[0]["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 72.5
    selected = ncu.select_kernel(rows, ["mock_kernel"])
    assert selected["Kernel Name"] == "mock_kernel"
    # 没有指定名字时跳过 at:: 内部 kernel
    assert ncu.select_kernel(rows)["Kernel Name"] == "mock_kernel"


def test_metrics_to_prompt() -> None:
    text = ncu.metrics_to_prompt({"Kernel Name": "k", "metric": 1.23456})
    assert "| metric | 1.2346 |" in text
    assert ncu.metrics_to_prompt({}).startswith("（NCU")
