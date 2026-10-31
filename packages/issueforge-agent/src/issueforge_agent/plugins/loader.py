from __future__ import annotations

import importlib
import logging
import os
from importlib import metadata
from typing import Any, Callable

from issueforge_agent.plugins.base import KernelGenerator
from issueforge_agent.plugins.llm import LlmGenerator

log = logging.getLogger("issueforge_agent.generator")

ENTRY_POINT_GROUP = "issueforge.kernel_generators"
BUILTIN_FACTORIES: dict[str, Callable[[], Any]] = {
    "llm": LlmGenerator,
}


class GeneratorError(RuntimeError):
    pass


def _load_object(spec: str) -> Any:
    """加载 ``module:attr`` 形式的对象。"""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise GeneratorError(f"插件路径必须形如 module:Class，收到：{spec}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise GeneratorError(f"无法导入插件模块 {module_name}：{exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise GeneratorError(f"模块 {module_name} 没有属性 {attr}") from exc


def _instantiate(factory: Any) -> KernelGenerator:
    # factory 可能是类（内置/entry point/模块属性）或已实例化的对象。
    if isinstance(factory, type):
        instance = factory()
    elif callable(factory) and not hasattr(factory, "generate"):
        instance = factory()
    else:
        instance = factory
    if not hasattr(instance, "generate"):
        raise GeneratorError("插件必须实现 generate(ctx) 方法")
    if not hasattr(instance, "diagnose"):
        # 允许极简插件只实现 generate
        from issueforge_agent.plugins.base import BaseGenerator

        class _Adapter(BaseGenerator):
            name = getattr(instance, "name", "plugin")

            def generate(self, ctx):  # type: ignore[no-untyped-def]
                return instance.generate(ctx)

        adapter = _Adapter()
        adapter.name = getattr(instance, "name", "plugin")
        return adapter
    return instance


def _entry_point_factories() -> dict[str, Any]:
    factories: dict[str, Any] = {}
    try:
        entries = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # Python < 3.10 兼容
        entries = metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for entry in entries:
        try:
            factories[entry.name] = entry.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("加载生成插件 entry point %s 失败：%s", entry.name, exc)
    return factories


def _env_factories() -> dict[str, Any]:
    raw = os.environ.get("ISSUEFORGE_GENERATOR_PLUGINS", "")
    factories: dict[str, Any] = {}
    for item in raw.split(","):
        spec = item.strip()
        if not spec:
            continue
        name = spec.rsplit(":", 1)[-1]
        try:
            factories[name] = _load_object(spec)
        except GeneratorError as exc:
            log.warning("%s", exc)
    return factories


def available_generators() -> dict[str, Any]:
    return {**BUILTIN_FACTORIES, **_entry_point_factories(), **_env_factories()}


def list_generators() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for name, factory in available_generators().items():
        target = getattr(factory, "__name__", type(factory).__name__)
        module = getattr(factory, "__module__", "")
        items.append({"name": name, "target": f"{module}:{target}".strip(":")})
    return items


def create_generator(name: str = "llm", plugin_spec: str | None = None) -> KernelGenerator:
    """按名称或显式 ``module:Class`` 创建生成插件。

    优先级：显式 plugin_spec > 环境/entry point 中同名插件 > 内置插件。
    """
    if plugin_spec:
        return _instantiate(_load_object(plugin_spec))
    factories = available_generators()
    if name not in factories:
        raise GeneratorError(
            f"未知生成插件：{name}。可用：{', '.join(sorted(factories)) or '（无）'}"
        )
    return _instantiate(factories[name])
