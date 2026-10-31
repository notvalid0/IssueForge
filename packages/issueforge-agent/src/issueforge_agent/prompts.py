"""向后兼容：内置生成插件的 prompt 模板已移动到 issueforge_agent.plugins.prompts。"""

from issueforge_agent.plugins.prompts import (  # noqa: F401
    OUTPUT_RULES,
    REFERENCE_RULES,
    build_diagnose_prompt,
    build_optimize_prompt,
    build_repair_prompt,
    build_seed_prompt,
)

__all__ = [
    "OUTPUT_RULES",
    "REFERENCE_RULES",
    "build_seed_prompt",
    "build_repair_prompt",
    "build_diagnose_prompt",
    "build_optimize_prompt",
]
