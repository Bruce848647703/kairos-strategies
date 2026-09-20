"""策略注册表：动态发现 kairos_strategies/channels/ 下所有 Strategy 子类。

新增一个渠道 = 往 channels/ 放一个模块，无需改动本文件（避免并行开发冲突）。
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, List

from . import channels
from .base import Strategy


def discover() -> List[Strategy]:
    """导入 channels 包下所有模块，收集 Strategy 子类并实例化（按 name 去重）。"""
    found: Dict[str, Strategy] = {}
    for mod_info in pkgutil.iter_modules(channels.__path__):
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(f".channels.{mod_info.name}", package=__package__)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, Strategy) or obj is Strategy:
                continue
            if obj.__module__ != mod.__name__:   # 只收集本模块定义的策略，避免重复导入
                continue
            name = getattr(obj, "name", "base")
            if not name or name == "base":
                continue
            try:
                found[name] = obj()
            except Exception:
                continue
    return list(found.values())


def by_channel(strategies: List[Strategy]) -> Dict[str, List[Strategy]]:
    """按 channel 分组，组内按 name 排序。"""
    grouped: Dict[str, List[Strategy]] = {}
    for s in strategies:
        grouped.setdefault(s.channel, []).append(s)
    for k in grouped:
        grouped[k].sort(key=lambda x: x.name)
    return dict(sorted(grouped.items()))
