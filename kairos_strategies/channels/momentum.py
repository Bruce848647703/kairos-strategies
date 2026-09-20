"""渠道：momentum —— 动量类策略（时序 / 截面 / 52周新高 / 双动量）。

收集途径：动量因子范式（time-series momentum、cross-sectional 12-1、
52-week-high 接近度、dual momentum）。全部为本仓库原创实现，仅用 numpy/pandas，
离线且确定性；第 t 期权重只使用截至 t 期（含）的价格信息，引擎再滞后一期，杜绝未来函数。

工程要点：
  - timing 类策略统一「等预算 1/N」，把 0~1 的持有强度缩放到组合层面最多满仓；
  - cross_section 类策略在被选中资产间等权归一，每行和为 1（无标的时为 0）；
  - 截面排名用 method='first' 打破并列，保证与列顺序一致的确定性结果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy
from .. import indicators as ind


def _equal_budget(strength: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """把 0~1 的持有强度缩放到等预算权重（每资产最多 1/N，组合最多满仓）。"""
    return strength.fillna(0.0) / max(data.n_assets, 1)


def _normalize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    """被选中资产等权归一：每行和为 1（该行无选中时保持全 0）。"""
    w = selected.astype(float)
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(row_sum, axis=0).fillna(0.0)


def _top_k_mask(factor: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    """按 factor 降序，每行选中前 ceil(n_valid*top_frac) 个（有有效值时至少 1 个）。

    返回布尔 DataFrame；并列按列顺序（method='first'）确定性打破，NaN 一律不选。
    """
    valid = factor.notna()
    n_valid = valid.sum(axis=1).to_numpy(dtype=float)                    # (n,)
    rank_desc = factor.rank(axis=1, ascending=False, method="first").to_numpy()  # (n, m)
    # 减去极小量再上取整，避免 n_valid*top_frac 恰为整数时的浮点上溢
    k = np.ceil(n_valid * float(top_frac) - 1e-9)
    k = np.where(n_valid > 0, np.maximum(k, 1.0), 0.0)                   # (n,)
    mask = (rank_desc <= k[:, None]) & valid.to_numpy()
    return pd.DataFrame(mask, index=factor.index, columns=factor.columns)


class TsMomentumStrategy(Strategy):
    name = "ts_momentum"
    channel = "momentum"
    universe = "timing"
    long_only = True
    description = "时序动量：过去 N 期累计收益为正则做多该资产、为负则空仓（等预算 1/N）。"
    hypothesis = "价格存在中期趋势延续（正自相关），顺势持有可获取动量溢价；在震荡反转或趋势急转时失效。"
    source = "动量范式——时序动量 time-series momentum（Moskowitz-Ooi-Pedersen 思路，原创实现）。"
    params = {"lookback": 60}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        mom = ind.momentum(data.prices, int(self.params["lookback"]))
        signal = (mom > 0).astype(float)
        signal[mom.isna()] = 0.0
        return _equal_budget(signal, data)


class XsMomentumStrategy(Strategy):
    name = "xs_momentum"
    channel = "momentum"
    universe = "cross_section"
    long_only = True
    description = "截面动量(12-1)：按剔除最近 skip 期的较长窗口收益做截面排名，做多排名前约 1/3 并等权归一。"
    hypothesis = "相对强势资产倾向继续跑赢（横截面动量）；剔除最近一期规避短期反转污染；在动量崩溃/风格急切换时失效。"
    source = "动量范式——横截面动量 / 12-1（Jegadeesh-Titman 思路，原创实现）。"
    params = {"lookback": 252, "skip": 21, "top_frac": 1.0 / 3.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        lookback = int(self.params["lookback"])
        skip = int(self.params["skip"])
        # 从 t-lookback 到 t-skip 的累计收益，仅用截至 t-skip 的历史价，绝不偷看未来
        sig = p.shift(skip) / p.shift(lookback) - 1.0
        selected = _top_k_mask(sig, self.params["top_frac"])
        return _normalize_selected(selected)


class High52wStrategy(Strategy):
    name = "high_52w"
    channel = "momentum"
    universe = "timing"
    long_only = True
    description = "52周新高接近度：价格越接近滚动 window 期最高价权重越高，低于阈值则空仓（等预算 1/N）。"
    hypothesis = "锚定效应使投资者不愿追高，接近历史高点的价格蕴含上行惯性；接近度捕捉突破动能；在假突破/见顶急跌时失效。"
    source = "动量范式——52周新高接近度 52-week-high proximity（George-Hwang 思路，原创实现）。"
    params = {"window": 250, "threshold": 0.90}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        window = int(self.params["window"])
        threshold = float(self.params["threshold"])
        roll_max = p.rolling(window, min_periods=window).max()   # 含当期，prox ≤ 1
        prox = p / roll_max
        denom = max(1.0 - threshold, 1e-12)
        strength = ((prox - threshold) / denom).clip(lower=0.0, upper=1.0).fillna(0.0)
        return _equal_budget(strength, data)


class DualMomentumStrategy(Strategy):
    name = "dual_momentum"
    channel = "momentum"
    universe = "cross_section"
    long_only = True
    description = "双动量：绝对动量(自身过去收益>0)过滤 + 相对动量(截面排名靠前)共同决定持有，被选中者等权归一。"
    hypothesis = "绝对动量规避下行资产（择大势），相对动量优选最强者，二者结合降低回撤；在动量崩溃/普跌无强势标的时失效。"
    source = "动量范式——双动量 dual momentum（Antonacci 思路，原创实现）。"
    params = {"lookback": 126, "top_frac": 1.0 / 3.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        mom = ind.momentum(data.prices, int(self.params["lookback"]))
        abs_ok = (mom > 0) & mom.notna()                 # 绝对动量：自身收益为正
        rel_ok = _top_k_mask(mom, self.params["top_frac"])  # 相对动量：截面靠前
        selected = abs_ok & rel_ok
        return _normalize_selected(selected)
