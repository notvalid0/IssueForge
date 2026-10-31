"""KernelBench 风格的示例任务：ReLU。

参考实现用于对比正确性与延迟；IssueForge 的 Kernel Agent 会生成 ModelNew。
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def get_inputs() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(4096, 4096, device="cuda", dtype=torch.float32)]


def get_init_inputs() -> list:
    return []


class ModelNew(nn.Module):
    """占位实现：真实运行时由 Agent 生成并替换。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
