from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


@dataclass(frozen=True)
class AgentPaths:
    data_dir: Path
    runs_dir: Path
    cache_dir: Path

    @classmethod
    def from_env(cls) -> "AgentPaths":
        data_dir = _expand(os.environ.get("ISSUEFORGE_AGENT_DATA_DIR", "~/.issueforge-agent"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data_dir,
            runs_dir=data_dir / "runs",
            cache_dir=data_dir / "cache",
        )


@dataclass(frozen=True)
class AgentConfig:
    """本地 Agent 配置。

    Server 只提供知识库、LLM 代理和运行记录；CUDA/NCU 全部在本地执行。
    """

    paths: AgentPaths
    server_url: str = "http://127.0.0.1:8796"
    server_token: str = ""
    backend: str = "mock"
    gpu: str = "Mock H100"
    ncu_bin: str = "ncu"
    cudaforge_root: str = ""
    max_rounds: int = 8
    generator: str = "llm"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        paths = AgentPaths.from_env()
        backend = os.environ.get("ISSUEFORGE_BACKEND", "mock")
        default_gpu = "Mock H100" if backend == "mock" else "Unknown GPU"
        return cls(
            paths=paths,
            server_url=os.environ.get("ISSUEFORGE_SERVER_URL", "http://127.0.0.1:8796").rstrip("/"),
            server_token=os.environ.get("ISSUEFORGE_AGENT_TOKEN", ""),
            backend=backend,
            gpu=os.environ.get("ISSUEFORGE_GPU", default_gpu),
            ncu_bin=os.environ.get("ISSUEFORGE_NCU_BIN", "ncu"),
            cudaforge_root=os.environ.get("ISSUEFORGE_CUDAFORGE_ROOT", ""),
            max_rounds=max(1, int(os.environ.get("ISSUEFORGE_MAX_ROUNDS", "8"))),
            generator=os.environ.get("ISSUEFORGE_GENERATOR", "llm"),
        )

    @property
    def server_configured(self) -> bool:
        return bool(self.server_url and self.server_token)

    def with_overrides(self, **kwargs: object) -> "AgentConfig":
        clean = {key: value for key, value in kwargs.items() if value not in (None, "")}
        return replace(self, **clean)


def detect_gpu() -> str:
    """尽量探测本机 GPU 名称；没有 torch/CUDA 时返回 Unknown GPU。"""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "Unknown GPU"
