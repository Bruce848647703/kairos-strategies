"""渠道：benchmark —— 基准策略，用于给其它策略提供对照。

收集途径：被动投资基准（买入持有）。
"""
from __future__ import annotations

import pandas as pd

from ..base import MarketData, Strategy


class EqualWeightBuyHold(Strategy):
    name = "equal_weight_buy_hold"
    channel = "benchmark"
    universe = "passive"
    long_only = True
    description = "等权买入并持有全部资产（不择时、不调仓）。"
    hypothesis = "被动持有获取市场 beta，作为主动策略的对照基准。"
    source = "被动投资基准——buy & hold。"
    params = {}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        w = 1.0 / max(data.n_assets, 1)
        return pd.DataFrame(w, index=data.dates, columns=data.symbols)
