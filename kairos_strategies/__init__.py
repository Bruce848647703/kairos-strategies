"""Kairos Strategies —— 自研策略研究库。

统一契约：策略输出「目标权重面板」，由向量化引擎回测，结果固化为研究记录(QR 流程)。
策略按「收集途径」分渠道组织于 kairos_strategies/channels/，registry 动态发现。
"""
from .base import MarketData, Strategy, align_weights
from .data import load_prices_csv, make_synthetic_universe
from .engine import Backtester, BacktestResult
from .registry import by_channel, discover
from . import report

__version__ = "0.1.0"

__all__ = [
    "MarketData", "Strategy", "align_weights",
    "make_synthetic_universe", "load_prices_csv",
    "Backtester", "BacktestResult",
    "discover", "by_channel", "report",
    "__version__",
]
