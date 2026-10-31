"""Kernel 生成插件层。

项目本体不内置具体生成方法：默认的 ``llm`` 插件也只是可替换的实现。
外部插件通过 entry point ``issueforge.kernel_generators`` 或
``ISSUEFORGE_GENERATOR_PLUGINS=module:Class`` 接入。
"""

from issueforge_agent.plugins.base import BaseGenerator, GenerationContext, KernelGenerator
from issueforge_agent.plugins.loader import (
    GeneratorError,
    available_generators,
    create_generator,
    list_generators,
)

__all__ = [
    "BaseGenerator",
    "GenerationContext",
    "KernelGenerator",
    "GeneratorError",
    "available_generators",
    "create_generator",
    "list_generators",
]
