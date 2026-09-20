"""渠道：microstructure —— 量价/微观结构策略（本仓库 100% 原创实现）。

收集途径：市场微观结构与量价分析范式（OBV 能量潮、VWAP 成交量加权均价、
量价背离/确认、流动性溢价），全部为原创代码，仅依赖 numpy/pandas 与自研
indicators，不引入任何第三方策略代码。

数据契约：优先使用 ``data.volumes``（成交量面板）。当 ``data.volumes`` 为
None（或缺失/全 NaN）时，各策略自动**优雅回退到纯价格代理逻辑**（详见各
类 docstring），保证不抛错且权重契约不变。

工程要点：
- 防未来函数：所有滚动量价统计只用「截至当期」的数据（新高判定等再
  shift 一期），引擎还会额外滞后一期，双重保险；
- timing 策略把 [0,1] 信号乘以等预算 1/N，组合毛敞口 ≤ 1；
  cross_section 策略逐行归一（行和 ≈ 1）；
- VWAP、归一化 OBV 量比、状态机持有、非流动性篮子加权等辅助函数全部
  在本模块内自实现，不 import 其它渠道的私有工具。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy


# ---------------------------------------------------------------------------
# 模块内自实现的小工具
# ---------------------------------------------------------------------------

def _get_volumes(data: MarketData) -> Optional[pd.DataFrame]:
    """取与价格对齐的成交量面板；缺失或全 NaN 时返回 None（触发纯价格回退）。"""
    v = data.volumes
    if v is None:
        return None
    v = v.reindex(index=data.prices.index, columns=data.prices.columns)
    v = v.apply(pd.to_numeric, errors="coerce")
    if not bool(v.notna().any().any()):
        return None
    return v.fillna(0.0).clip(lower=0.0)


def _rolling_vwap(prices: pd.DataFrame, volumes: pd.DataFrame,
                  window: int) -> pd.DataFrame:
    """滚动成交量加权均价 VWAP = Σ(price×vol)/Σvol（窗口含当期，不看未来）。"""
    pv = (prices * volumes).rolling(window, min_periods=window).sum()
    vv = volumes.rolling(window, min_periods=window).sum()
    return pv / vv.replace(0.0, np.nan)


def _volume_imbalance_ratio(prices: pd.DataFrame, volumes: pd.DataFrame,
                            window: int) -> pd.DataFrame:
    """归一化 OBV 量比：滚动窗口内 (上涨日成交量-下跌日成交量)/总成交量 ∈ [-1,1]。"""
    ret = prices.pct_change()
    up = volumes.where(ret > 0, 0.0).rolling(window, min_periods=window).sum()
    dn = volumes.where(ret < 0, 0.0).rolling(window, min_periods=window).sum()
    return (up - dn) / (up + dn).replace(0.0, np.nan)


def _price_imbalance_ratio(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """纯价格回退版失衡比：(上涨天数-下跌天数)/总天数 ∈ [-1,1]。"""
    ret = prices.pct_change()
    up = (ret > 0).astype(float).rolling(window, min_periods=window).sum()
    dn = (ret < 0).astype(float).rolling(window, min_periods=window).sum()
    return (up - dn) / (up + dn).replace(0.0, np.nan)


def _stateful_long(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """状态机：enter 置 1 并持有，exit_ 清 0，其间保持（双阈值滞后带效果）。"""
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


def _to_weights(signal: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """把 [0,1] 信号转成等预算权重（每资产 1/N，组合毛敞口 ≤ 1）。"""
    s = signal.fillna(0.0).clip(lower=0.0, upper=1.0)
    return s / max(data.n_assets, 1)


def _illiquid_basket_weights(score: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    """把「越高越非流动（越好）」的截面分数转成只做多、逐行归一的权重面板。

    步骤（逐行/逐日，仅用当期截面信息）：
      1. 分数降序排名，选中前 ``k = ceil(top_frac * N)`` 名做多，其余为 0；
         NaN 分数（预热期）一律不选；
      2. 选中集内按「调和倾斜」加权 ``raw = 1/rank``（第 1 名权重最大，
         第 2 名减半……），对分数离群值天然稳健；
      3. 逐行归一 ``w = raw / Σraw``，选中行权重和 ≈ 1；预热行整行为 0。
    """
    score = score.replace([np.inf, -np.inf], np.nan)
    n = score.shape[1]
    k = max(1, int(np.ceil(top_frac * n - 1e-9)))
    ranks = score.rank(axis=1, ascending=False, method="min", na_option="keep")
    sel = (ranks <= k) & score.notna()
    raw = (1.0 / ranks).where(sel, 0.0)
    row_sum = raw.sum(axis=1)
    w = raw.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)
    return w.clip(lower=0.0, upper=1.0)


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------

class VolumeImbalanceStrategy(Strategy):
    """量价失衡择时（归一化 OBV）。

    滚动窗口内「上涨日成交量 vs 下跌日成交量」的归一化比率衡量买卖净压：
    比率突破正阈值（买压主导）做多并持有，跌破负阈值（卖压主导）离场。

    回退：``data.volumes`` 为 None 时退化为纯价格逻辑——用「上涨天数 vs
    下跌天数」比率代替量能比率（等价于成交量恒为 1 的 OBV）。
    """

    name = "volume_imbalance"
    channel = "microstructure"
    universe = "timing"
    long_only = True
    description = "归一化 OBV 量价失衡：滚动窗口内上涨日成交量相对下跌日成交量占优（买压强）做多，卖压占优离场。"
    hypothesis = ("成交量是价格运动的『确认票』：放量上涨说明知情买盘主导、趋势更可能延续，"
                  "放量下跌或缩量上涨则预示动能衰竭。归一化的上/下行量比剔除了资产间量能规模差异，"
                  "能刻画买卖净压方向。当市场处于无量单边行情、量能结构被外生事件（如指数调仓）"
                  "扰动，或失衡比在阈值附近反复摆动时，信号会钝化并产生换手损耗。")
    source = "微观结构/量价分析——OBV（能量潮）思想的原创归一化实现。"
    params = {"window": 20, "enter": 0.2, "exit": -0.2}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        w = self.params["window"]
        v = _get_volumes(data)
        ratio = (_volume_imbalance_ratio(data.prices, v, w) if v is not None
                 else _price_imbalance_ratio(data.prices, w))
        enter = ratio > self.params["enter"]
        exit_ = ratio < self.params["exit"]
        return _to_weights(_stateful_long(enter, exit_), data)


class VwapReversionStrategy(Strategy):
    """VWAP 均值回归择时。

    滚动成交量加权均价 VWAP = Σ(price×vol)/Σvol 代表近期市场的平均持仓
    成本。价格显著低于 VWAP（折价超过进入带）视为超卖做多，回到/高于
    VWAP 即离场兑现（双阈值滞后带 + 状态机持有，抑制日频抖动）。

    回退：``data.volumes`` 为 None 时用等权滚动均值 SMA 代替 VWAP 作为
    成本锚，其余逻辑不变（纯价格均值回归）。
    """

    name = "vwap_reversion"
    channel = "microstructure"
    universe = "timing"
    long_only = True
    description = "VWAP 回归：价格显著低于滚动成交量加权均价做多，回到/高于 VWAP 离场（均值回归）。"
    hypothesis = ("VWAP 是近期成交量的加权平均成本，代表市场共识价位：价格大幅折价于 VWAP 时，"
                  "流动性提供者与价值买盘倾向入场修复偏离；量能加权的锚比等权均线更贴近真实换手"
                  "成本，回归信号更可靠。当折价由持续基本面利空驱动（趋势性下跌）而非流动性冲击时，"
                  "『接飞刀』会放大亏损；高成交量把 VWAP 快速拉向现价时信号也会变钝。")
    source = "微观结构/量价分析——机构执行算法常用的 VWAP 锚，原创均值回归化实现。"
    params = {"window": 20, "enter_below": -0.02, "exit_above": 0.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        w = self.params["window"]
        p = data.prices
        v = _get_volumes(data)
        anchor = _rolling_vwap(p, v, w) if v is not None else ind.sma(p, w)
        dev = p / anchor - 1.0
        enter = (dev < self.params["enter_below"]) & anchor.notna()
        exit_ = (dev >= self.params["exit_above"]) & anchor.notna()
        return _to_weights(_stateful_long(enter, exit_), data)


class VolumePriceDivergenceStrategy(Strategy):
    """量价背离/确认择时（连续权重版）。

    价格上行（mom_window 期收益为正）时持有，仓位大小由「量能确认度」
    连续调节：当前成交量相对自身滚动均值的量比越高（价涨量增，确认），
    权重越大；若价格创 N 期新高但量比低于背离阈值（价涨量缩，背离），
    视为动能衰竭强制清仓离场。

    回退：``data.volumes`` 为 None 时退化为纯价格动量逻辑——收益为正则
    满预算持有（量能确认度恒为 1，且不做背离清仓）。
    """

    name = "volume_price_divergence"
    channel = "microstructure"
    universe = "timing"
    long_only = True
    description = "量价背离：价涨量增（确认）按量能连续加权做多，价创新高但量缩（背离）清仓离场。"
    hypothesis = ("健康的上涨需要成交量确认：放量新高意味着新增资金入场、信息被广泛接受，趋势"
                  "延续概率高；缩量新高说明追买意愿枯竭，是经典的顶部背离信号。按量比连续调仓"
                  "（而非 0/1 开关）能让仓位与信号置信度成正比，降低阈值敏感性。当量能季节性"
                  "萎缩（假期）、或上涨由低量能的逼空行情驱动时，确认/背离判别可能失真。")
    source = "微观结构/量价分析——经典『量在价先/背离』范式的原创连续权重实现。"
    params = {"mom_window": 20, "vol_window": 20, "high_window": 20,
              "vr_floor": 0.5, "vr_cap": 1.5, "div_ratio": 0.9}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        mom = p.pct_change(self.params["mom_window"])
        v = _get_volumes(data)
        if v is not None:
            base = v.rolling(self.params["vol_window"],
                             min_periods=self.params["vol_window"]).mean()
            vr = v / base.replace(0.0, np.nan)          # 量比：现量/自身均量
            span = self.params["vr_cap"] - self.params["vr_floor"]
            vol_factor = ((vr - self.params["vr_floor"]) / span).clip(0.0, 1.0)
            prev_max = p.rolling(self.params["high_window"],
                                 min_periods=self.params["high_window"]).max().shift(1)
            new_high = p > prev_max                      # 创 N 期新高（不含当日）
            divergence = new_high & (vr < self.params["div_ratio"])
            s = ((mom > 0).astype(float) * vol_factor).where(~divergence, 0.0)
        else:
            s = (mom > 0).astype(float)                  # 纯价格回退：动量持有
        s = s.where(mom.notna(), 0.0).fillna(0.0)
        return _to_weights(s, data)


class LiquidityPremiumStrategy(Strategy):
    """流动性溢价截面策略。

    非流动性打分（越高越『冷清』越好）由两个截面分位合成：
      1. 量能水平：滚动 fast 期均量在截面中的分位（越低越非流动）；
      2. 相对换手：fast 期均量相对自身 slow 期均量之比（缩量越明显越非流动）。
    做多得分最高的前 top_frac 篮子，选中集内按 1/rank 调和倾斜加权，
    逐行归一到和 ≈ 1；预热期（信息不足）整行为 0。

    回退：``data.volumes`` 为 None 时用价格已实现波动作为『活跃度』代理
    （低波动/低活跃 ≈ 低流动性），打分与加权逻辑不变。
    """

    name = "liquidity_premium"
    channel = "microstructure"
    universe = "cross_section"
    long_only = True
    description = "流动性溢价：偏好低流动性/低换手（滚动成交量相对自身均值偏低且截面量能冷清）的资产，做多最非流动篮子。"
    hypothesis = ("流动性是资产定价的系统性维度：持有『冷清、难变现』的资产需要补偿，长期看"
                  "低换手/低量能组合的期望收益高于高流动性组合（Amihud 式非流动性溢价）；同时"
                  "量能相对自身均值萎缩往往处于关注度低谷，后续关注度修复带来超额收益。当流动性"
                  "危机中非流动资产折价加剧且无法变现、或溢价被拥挤套利压缩时，策略会失效。")
    source = "微观结构/学术因子——流动性溢价（Amihud 非流动性）思想的原创量价实现。"
    params = {"fast": 10, "slow": 60, "top_frac": 1.0 / 3.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        f, sl = self.params["fast"], self.params["slow"]
        v = _get_volumes(data)
        if v is not None:
            recent = v.rolling(f, min_periods=f).mean()
            base = v.rolling(sl, min_periods=sl).mean()
        else:                                        # 纯价格回退：活跃度=波动
            r = data.returns()
            recent = r.rolling(f, min_periods=f).std()
            base = r.rolling(sl, min_periods=sl).std()
        rel = recent / base.replace(0.0, np.nan)     # 相对自身均值的量能/活跃度
        lvl_rank = ind.cross_sectional_rank(recent, ascending=True)
        rel_rank = ind.cross_sectional_rank(rel, ascending=True)
        score = -(lvl_rank + rel_rank)               # 越冷清分数越高
        return _illiquid_basket_weights(score, self.params["top_frac"])
