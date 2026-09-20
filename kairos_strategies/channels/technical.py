"""渠道：technical —— 经典技术分析择时策略（逐资产 timing，只做多）。

收集途径：经典技术分析范式（均线、RSI、MACD、布林带、唐奇安通道/海龟）。
全部为本仓库原创实现，输出统一为目标权重面板。

工程要点：所有阈值信号均加「滞后带 (hysteresis)」并以状态机持有，
进入/退出用不同阈值，显著降低日频 whipsaw 带来的换手与成本拖累。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy
from .. import indicators as ind


def _to_weights(signal: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """把 0/1 持有状态转成等预算权重（每资产 1/N，组合最大满仓）。"""
    return signal.fillna(0.0) / max(data.n_assets, 1)


def _stateful_long(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """状态机：enter 置 1 并持有，exit 清 0，其间保持（含滞后带效果）。"""
    E = enter.fillna(False).values.astype(bool)
    X = exit_.fillna(False).values.astype(bool)
    out = np.zeros(E.shape, dtype=float)
    hold = np.zeros(E.shape[1], dtype=float)
    for t in range(E.shape[0]):
        for j in range(E.shape[1]):
            if X[t, j]:
                hold[j] = 0.0
            elif E[t, j]:
                hold[j] = 1.0
        out[t] = hold
    return pd.DataFrame(out, index=enter.index, columns=enter.columns)


class SmaCrossStrategy(Strategy):
    name = "sma_cross"
    channel = "technical"
    universe = "timing"
    long_only = True
    description = "双均线 + 滞后带：短均线带缓冲上穿长均线做多，反向跌破离场。"
    hypothesis = "价格存在中期趋势，带缓冲的金叉能过滤噪声、捕捉趋势启动。"
    source = "经典技术分析——移动平均交叉（moving average crossover）。"
    params = {"fast": 10, "slow": 30, "band": 0.01}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        b = self.params["band"]
        fast = ind.sma(p, self.params["fast"])
        slow = ind.sma(p, self.params["slow"])
        enter = fast > slow * (1 + b)
        exit_ = fast < slow * (1 - b)
        enter[fast.isna() | slow.isna()] = False
        exit_[fast.isna() | slow.isna()] = False
        return _to_weights(_stateful_long(enter, exit_), data)


class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"
    channel = "technical"
    universe = "timing"
    long_only = True
    description = "RSI 均值回归：超卖买入，回升至中性上方离场（双阈值滞后）。"
    hypothesis = "短期超跌后价格倾向反弹；及时止盈避免持有下行趋势。"
    source = "经典技术分析——RSI 超买超卖。"
    params = {"window": 14, "enter_below": 35.0, "exit_above": 55.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        r = ind.rsi(data.prices, self.params["window"])
        enter = r < self.params["enter_below"]
        exit_ = r > self.params["exit_above"]
        enter[r.isna()] = False
        exit_[r.isna()] = False
        return _to_weights(_stateful_long(enter, exit_), data)


class MacdTrendStrategy(Strategy):
    name = "macd_trend"
    channel = "technical"
    universe = "timing"
    long_only = True
    description = "MACD 动能：柱状图转正做多，明显转负离场（带缓冲）。"
    hypothesis = "MACD 反映快慢动能差，柱持续为正代表上升动能占优。"
    source = "经典技术分析——MACD。"
    params = {"fast": 12, "slow": 26, "signal": 9, "band": 0.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        macd_line, signal_line, hist = ind.macd(
            data.prices, self.params["fast"], self.params["slow"], self.params["signal"])
        b = self.params["band"]
        enter = hist > b
        exit_ = hist < -b
        enter[macd_line.isna() | signal_line.isna()] = False
        exit_[macd_line.isna() | signal_line.isna()] = False
        return _to_weights(_stateful_long(enter, exit_), data)


class BollingerBreakoutStrategy(Strategy):
    name = "bollinger_breakout"
    channel = "technical"
    universe = "timing"
    long_only = True
    description = "布林带突破：收盘上穿上轨做多，回落至中轨下方离场。"
    hypothesis = "突破上轨代表强势启动，趋势延续至均值附近。"
    source = "经典技术分析——布林带突破。"
    params = {"window": 20, "num_std": 2.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.prices
        mid, upper, _ = ind.bollinger(close, self.params["window"], self.params["num_std"])
        enter = close > upper.shift(1)
        exit_ = close < mid.shift(1)
        return _to_weights(_stateful_long(enter, exit_), data)


class DonchianTurtleStrategy(Strategy):
    name = "donchian_turtle"
    channel = "technical"
    universe = "timing"
    long_only = True
    description = "唐奇安通道突破（海龟）：创 N 日新高做多，跌破 M 日新低离场。"
    hypothesis = "区间突破后趋势延续（海龟交易法则）。"
    source = "经典技术分析——唐奇安通道 / 海龟交易法。"
    params = {"entry": 20, "exit": 10}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.prices
        upper = close.rolling(self.params["entry"], min_periods=self.params["entry"]).max().shift(1)
        lower = close.rolling(self.params["exit"], min_periods=self.params["exit"]).min().shift(1)
        enter = close > upper
        exit_ = close < lower
        return _to_weights(_stateful_long(enter, exit_), data)
