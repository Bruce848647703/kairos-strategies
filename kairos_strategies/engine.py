"""向量化回测引擎（自包含）。

时序约定：第 t 期决定的目标权重在 t→t+1 期间持有，组合收益用「上一期权重 × 本期资产收益」，
杜绝未来函数。换手按「目标权重 vs 漂移后持有权重」的绝对变化计，并乘以单边成本率扣减。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import metrics
from .base import MarketData, align_weights


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    cost_rate: float
    _metrics: Dict[str, float]

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics


class Backtester:
    def __init__(self, cost_rate: float = 0.001, risk_free: float = 0.0,
                 periods_per_year: int = 252):
        self.cost_rate = float(cost_rate)
        self.risk_free = float(risk_free)
        self.periods_per_year = int(periods_per_year)

    def run(self, data: MarketData, weights: pd.DataFrame) -> BacktestResult:
        prices = data.prices.astype("float64").sort_index()
        w = align_weights(weights, MarketData(prices, periods_per_year=data.periods_per_year),
                          clip=1.0)
        asset_ret = prices.pct_change().fillna(0.0)
        held = w.shift(1).fillna(0.0)                 # 实际持有 = 上一期目标
        gross = (held * asset_ret).sum(axis=1)

        drifted = self._drift(held, asset_ret)
        turnover = (w - drifted).abs().sum(axis=1)
        turnover.iloc[0] = float(w.iloc[0].abs().sum())  # 建仓换手

        costs = turnover * self.cost_rate
        net = gross - costs
        equity = (1.0 + net).cumprod()
        m = metrics.summarize(net, turnover, self.risk_free, self.periods_per_year)
        return BacktestResult(returns=net, gross_returns=gross, equity=equity,
                              turnover=turnover, weights=held,
                              cost_rate=self.cost_rate, _metrics=m)

    @staticmethod
    def _drift(held: pd.DataFrame, asset_ret: pd.DataFrame) -> pd.DataFrame:
        """持有权重随收益漂移后的下一期权重（占净值比例，用于计算真实换手）。

        组合净值因子 = 1 + r_p，其中 r_p = Σ(持有_i × 收益_i)（现金部分收益为 0）。
        用 1+r_p 作分母对任意敞口都成立：满仓(Σ=1)、部分持仓(含现金)、
        乃至美元中性(Σ=0) 都不会出现除零或换手虚增。
        """
        grown = held * (1.0 + asset_ret)
        r_p = (held * asset_ret).sum(axis=1)
        denom = (1.0 + r_p).replace(0.0, np.nan)
        return grown.div(denom, axis=0).fillna(0.0)
