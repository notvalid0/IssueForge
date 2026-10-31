from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from issueforge_core import ncu
from issueforge_agent.kernel.mock import MockBackend

_KERNEL_RE = re.compile(r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)")


def extract_cuda_kernel_names(code: str) -> list[str]:
    return list(dict.fromkeys(_KERNEL_RE.findall(code or "")))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CudaForgeBackend:
    """CUDA C++ 后端：复用 CudaForge 的 compare_and_bench，并用 NCU 采集硬件指标。"""

    name = "cudaforge"

    def __init__(self, cudaforge_root: str | Path, ncu_bin: str = "ncu", gpu: str = "Unknown GPU"):
        self.root = Path(cudaforge_root).expanduser().resolve()
        self.ncu_bin = ncu_bin
        self.gpu = gpu
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

    def _torch(self):
        import torch  # type: ignore

        return torch

    def available(self) -> tuple[bool, str]:
        try:
            torch = self._torch()
        except Exception as exc:  # noqa: BLE001
            return False, f"torch 不可用：{exc}"
        if not torch.cuda.is_available():
            return False, "torch.cuda 不可用"
        if not (self.root / "utils" / "compile_and_run.py").exists():
            return False, f"找不到 CudaForge：{self.root}"
        return True, ""

    def prepare(self, task_path: Path) -> dict:
        text = task_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "task_name": task_path.stem,
            "reference": text,
            "gpu": self.gpu,
            "backend": self.name,
            "kernel_names": extract_cuda_kernel_names(text),
        }

    def generate(self, prompt: str, phase: str, round_index: int, current_code: str = "") -> str:
        # 真正的生成由 Agent 的 LLM 完成；这里只作为没有 LLM 时的兜底。
        return MockBackend(gpu=self.gpu).generate(prompt, phase, round_index, current_code)

    def evaluate(self, code: str, task_path: Path, round_index: int, work_dir: Path | None = None) -> dict:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        from utils.compile_and_run import AccuracyError, CompilationError, compare_and_bench  # type: ignore

        work_dir = work_dir or Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)
        candidate = work_dir / f"candidate_round_{round_index}.py"
        candidate.write_text(code, encoding="utf-8")
        try:
            metrics = compare_and_bench(
                ref_py=task_path,
                test_py=candidate,
                device_idx=0,
                warmup=5,
                repeat=20,
                tol=1e-3,
            )
            metrics["runnable"] = True
            ref_avg = float(metrics["ref_latency_ms"]["avg"])
            test_avg = float(metrics["test_latency_ms"]["avg"])
            metrics["score"] = ref_avg / max(1e-9, test_avg)
            return metrics
        except CompilationError as exc:
            return {"runnable": False, "score": None, "error_type": "CompilationError", "message": str(exc)[-4000:]}
        except AccuracyError as exc:
            return {"runnable": False, "score": None, "error_type": "AccuracyError", "message": str(exc)[-4000:]}
        except Exception as exc:  # noqa: BLE001
            return {
                "runnable": False,
                "score": None,
                "error_type": type(exc).__name__,
                "message": str(exc)[-4000:],
            }

    def _bench_script(self, candidate: Path, task_path: Path, out: Path) -> Path:
        script = f'''import importlib.util, sys
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ref = load({str(task_path)!r}, "issueforge_ref")
cand = load({str(candidate)!r}, "issueforge_cand")
init = ref.get_init_inputs() if hasattr(ref, "get_init_inputs") else []
inputs = ref.get_inputs() if hasattr(ref, "get_inputs") else []
inputs = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
model = ref.Model(*init).cuda().eval()
new = cand.ModelNew(*init).cuda().eval()
with torch.no_grad():
    model(*inputs)
    for _ in range(5):
        new(*inputs)
torch.cuda.synchronize()
'''
        out.write_text(script, encoding="utf-8")
        return out

    def profile(self, code: str, task_path: Path, round_index: int, speedup: float = 1.0, work_dir: Path | None = None) -> dict:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        work_dir = work_dir or Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)
        candidate = work_dir / f"candidate_round_{round_index}.py"
        candidate.write_text(code, encoding="utf-8")
        bench = self._bench_script(candidate, task_path, work_dir / f"bench_round_{round_index}.py")
        csv_path = work_dir / f"ncu_round_{round_index}.csv"
        ncu.profile(
            bench,
            csv_path,
            ncu_bin=self.ncu_bin,
            kernel_names=extract_cuda_kernel_names(code),
        )
        rows = ncu.parse_ncu_csv(csv_path)
        selected = ncu.select_kernel(rows, extract_cuda_kernel_names(code))
        return selected

    def health(self) -> dict:
        ok, reason = self.available()
        return {
            "ok": ok,
            "backend": self.name,
            "gpu": self.gpu,
            "ncu": ncu.ncu_version(self.ncu_bin),
            "ncu_available": ncu.ncu_available(self.ncu_bin),
            "cudaforge_root": str(self.root),
            "reason": reason,
        }
