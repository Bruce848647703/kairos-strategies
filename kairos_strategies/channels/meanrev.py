"""渠道：meanrev —— 均值回归 / 统计套利策略（可多空，市场中性配对）。

收集途径：统计套利经典范式——时序 z-score 回归、布林带回归、均线偏离回归、
配对价差（对数价差 z-score）多空。全部为本仓库原创实现，输出统一为目标权重面板。

工程要点：
1. 三个择时策略共用「三态状态机 + 滞后带」：进入阈值（极端偏离）与退出阈值
   （回到中枢附近）分离，避免在阈值附近反复翻转产生高换手；状态用前向填充保持。
2. 择时信号 ∈ {-1, 0, +1}，统一乘以 1/N 做等预算分配，保证每行绝对值之和 ≤ 1。
3. 配对价差为市场中性：两腿等权反向，每行权重之和恒为 0，绝对值之和 ≤ 1。
4. 只用截至当期的滚动窗口（trailing window），不含任何未来信息；引擎还会再滞后一期。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy
from .. import indicators as ind

_EPS = 1e-9          # 价格下限保护，避免 log(0)
_POS_EPS = 1e-12     # 除零保护阈值


def _safe_prices(data: MarketData) -> pd.DataFrame:
    """价格面板转 float64 并裁到正数（对数/比值运算的安全前提）。"""
    return data.prices.astype("float64").clip(lower=_EPS)


def _budget(state: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """把 {-1, 0, +1} 状态面板转成等预算权重（每资产 1/N，行绝对值和 ≤ 1）。"""
    n = max(int(data.n_assets), 1)
    out = (state.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0))
    return out / float(n)


def _stateful_position(long_on: pd.DataFrame, short_on: pd.DataFrame,
                       flat_on: pd.DataFrame) -> pd.DataFrame:
    """三态状态机：把触发点标记为 +1/-1/0，其余置 NaN 后前向填充以保持持仓。

    优先级：平仓 > 开多 > 开空（同一 bar 触发多个时，先降风险再定方向）。
    未触发任何条件时沿用上一状态，这就是滞后带（hysteresis）的实现方式。
    """
    L = long_on.fillna(False).values.astype(bool)
    S = short_on.fillna(False).values.astype(bool)
    F = flat_on.fillna(False).values.astype(bool)
    marks = np.where(F, 0.0, np.where(L, 1.0, np.where(S, -1.0, np.nan)))
    state = pd.DataFrame(marks, index=long_on.index, columns=long_on.columns)
    return state.ffill().fillna(0.0)


def _three_bands(dev: pd.DataFrame, entry: float, exit_: float) -> pd.DataFrame:
    """由「偏离度面板」生成三态持仓：dev < -entry 做多、dev > +entry 做空、|dev| <= exit 平仓。"""
    long_on = dev < -entry
    short_on = dev > entry
    flat_on = dev.abs() <= exit_
    long_on = long_on.fillna(False) & dev.notna()
    short_on = short_on.fillna(False) & dev.notna()
    flat_on = flat_on.fillna(False) & dev.notna()
    return _stateful_position(long_on, short_on, flat_on)


def _band_deviation(close: pd.DataFrame, window: int, num_std: float) -> pd.DataFrame:
    """布林归一偏离：(close - mid) / 半带宽。> 1 升破上轨，< -1 跌破下轨，0 位于中轨。"""
    mid, upper, lower = ind.bollinger(close, window, num_std)
    half = (upper - lower) / 2.0                      # = num_std * 滚动标准差
    return (close - mid) / half.where(half > _POS_EPS)


class ZscoreReversionStrategy(Strategy):
    name = "zscore_reversion"
    channel = "meanrev"
    universe = "timing"
    long_only = False
    description = "滚动 z-score 回归：对数价格相对 N 期均值偏离 z < -阈值做多、z > +阈值做空，|z| 收敛到中枢附近平仓（三态滞后带）。"
    hypothesis = "核心假设：无漂移或弱漂移品种的价格围绕中枢震荡，标准化偏离越极端、回归概率越高，故反向持仓期望为正。失效场景：强趋势/结构性位移（如趋势品种持续单边），z 长期停留在极端区，逆势持仓被反复止损。"
    source = "统计套利经典范式——时序 z-score 均值回归（本仓库原创实现，指标复用自研 indicators.rolling_zscore）。"
    params = {"window": 20, "entry_z": 1.5, "exit_z": 0.5}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        logp = np.log(_safe_prices(data))
        z = ind.rolling_zscore(logp, int(self.params["window"]))
        state = _three_bands(z, float(self.params["entry_z"]), float(self.params["exit_z"]))
        return _budget(state, data)


class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"
    channel = "meanrev"
    universe = "timing"
    long_only = False
    description = "布林带回归：收盘跌破下轨做多、升破上轨做空，价格回到中轨附近（带内 exit_band 比例）双向平仓。"
    hypothesis = "核心假设：布林带刻画了近期价格的常态波动区间，触及边界多为流动性冲击/过度反应，价格倾向回到中轨。失效场景：真突破行情（波动率 regime 切换）中带轨会被持续「骑乘」，逆势持仓亏损；带宽收窄期信号稀疏。"
    source = "经典技术分析——布林带均值回归（与 technical 渠道的 bollinger_breakout 突破逻辑相反，本仓库独立原创实现）。"
    params = {"window": 20, "num_std": 2.0, "entry_band": 1.0, "exit_band": 0.2}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        close = _safe_prices(data)
        dev = _band_deviation(close, int(self.params["window"]), float(self.params["num_std"]))
        state = _three_bands(dev, float(self.params["entry_band"]), float(self.params["exit_band"]))
        return _budget(state, data)


class MaDeviationStrategy(Strategy):
    name = "ma_deviation"
    channel = "meanrev"
    universe = "timing"
    long_only = False
    description = "均线偏离回归：价格相对 N 期均线的百分比偏离显著为负做多、显著为正做空，偏离收敛回均线附近（滞后带）平仓以降低换手。"
    hypothesis = "核心假设：均线代表近期持仓成本中枢，价格远离成本中枢后存在回归拉力（过度反应修正）。失效场景：单边趋势中偏离会长期维持在同一侧（均线追随），策略逆势加仓并承受回撤；低波动期偏离不足，几乎不交易。"
    source = "经典技术分析——均线乖离率(BIAS)回归，本仓库原创实现（指标复用自研 indicators.sma）。"
    params = {"window": 20, "entry_dev": 0.05, "exit_dev": 0.015}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        close = _safe_prices(data)
        ma = ind.sma(close, int(self.params["window"]))
        dev = close / ma.where(ma > _POS_EPS) - 1.0
        state = _three_bands(dev, float(self.params["entry_dev"]), float(self.params["exit_dev"]))
        return _budget(state, data)


class PairsSpreadStrategy(Strategy):
    name = "pairs_spread"
    channel = "meanrev"
    universe = "cross_section"
    long_only = False
    description = "配对价差统计套利：取两只资产的对数价差滚动 z-score，价差被低估时做多 A/做空 B、被高估时反向，回归中枢平仓；两腿等权，组合市场中性。"
    hypothesis = "核心假设：同族资产的对数价差（相对估值）比绝对价格更平稳，价差极端偏离后倾向收敛，多空对冲可剥离市场 beta。失效场景：两腿基本面脱钩（协整关系破裂）时价差不回归而是持续发散，亏损无自然上限；配对选择错误是最大风险来源。"
    source = "统计套利经典范式——pairs trading / 价差 z-score 多空（本仓库原创实现）。"
    params = {"leg_a": "A0", "leg_b": "A1", "window": 40,
              "entry_z": 1.5, "exit_z": 0.5, "leg_weight": 0.5}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        leg_a, leg_b = self._pick_legs(list(data.symbols))
        if leg_a is None or leg_b is None:
            return w                                  # 不足两只资产：无法配对，空仓
        close = _safe_prices(data)
        spread = np.log(close[leg_a]) - np.log(close[leg_b])   # 对数价差（等价于价格比的取对数）
        z = ind.rolling_zscore(spread.to_frame("spread"),
                              int(self.params["window"]))["spread"]
        state = _three_bands(z.to_frame("spread"),
                            float(self.params["entry_z"]),
                            float(self.params["exit_z"]))["spread"]
        # state = +1 -> 价差偏低（A 相对 B 被低估）：多 A 空 B；-1 反之
        leg_w = float(self.params["leg_weight"]) * state.reindex(index=data.dates).fillna(0.0)
        w[leg_a] = leg_w
        w[leg_b] = -leg_w
        return w

    def _pick_legs(self, symbols: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """确定配对两腿：优先用 params 指定的代码，缺失则退回前两列（顺序确定、可复现）。"""
        if len(symbols) < 2:
            return None, None
        a, b = str(self.params["leg_a"]), str(self.params["leg_b"])
        if a in symbols and b in symbols and a != b:
            return a, b
        return symbols[0], symbols[1]
