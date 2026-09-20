"""自研技术指标库（纯 numpy/pandas，逐列作用于价格面板或序列）。

所有函数对 DataFrame 按列独立计算，对 Series 直接计算；不引入未来数据。
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

Num = Union[pd.Series, pd.DataFrame]


def sma(x: Num, window: int) -> Num:
    """简单移动平均。"""
    return x.rolling(window, min_periods=window).mean()


def ema(x: Num, span: int) -> Num:
    """指数移动平均。"""
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: Num, window: int = 14) -> Num:
    """相对强弱指标 RSI（Wilder 平滑），取值 0~100。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def macd(close: Num, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD：返回 (macd_line, signal_line, hist)。"""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: Num, window: int = 20, num_std: float = 2.0):
    """布林带：返回 (mid, upper, lower)。"""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return mid, mid + num_std * std, mid - num_std * std


def donchian(high: Num, low: Num, window: int):
    """唐奇安通道：返回 (upper=滚动最高, lower=滚动最低, mid)。"""
    upper = high.rolling(window, min_periods=window).max()
    lower = low.rolling(window, min_periods=window).min()
    return upper, lower, (upper + lower) / 2.0


def true_range(high: Num, low: Num, close: Num) -> Num:
    """真实波幅 TR。"""
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1) if a.ndim == 1 else \
        np.maximum.reduce([a.values, b.values, c.values])


def atr(high: Num, low: Num, close: Num, window: int = 14) -> Num:
    """平均真实波幅 ATR（Wilder 平滑）。"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_zscore(x: Num, window: int) -> Num:
    """滚动 z-score（当前值相对过去 window 期均值/标准差）。"""
    m = x.rolling(window, min_periods=window).mean()
    s = x.rolling(window, min_periods=window).std()
    return (x - m) / s.replace(0.0, np.nan)


def momentum(close: Num, window: int) -> Num:
    """动量：过去 window 期收益率。"""
    return close.pct_change(window)


def realized_vol(close: Num, window: int = 20, periods_per_year: int = 252) -> Num:
    """已实现波动率（年化）。"""
    return close.pct_change().rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)


def cross_sectional_rank(x: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """截面排名归一到 [0,1]（按行）。"""
    return x.rank(axis=1, ascending=ascending, pct=True)


def ts_standardize(x: Num, window: int) -> Num:
    """时序标准化（滚动 z-score 的别名，语义更清晰）。"""
    return rolling_zscore(x, window)
