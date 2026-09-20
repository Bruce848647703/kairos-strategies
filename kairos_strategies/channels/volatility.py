"""渠道：volatility —— 以「波动率」为核心信息的风险预算与突破择时策略（只做多）。

收集途径：波动率目标 (volatility targeting)、ATR 通道突破、波动率 regime 过滤、
时序动量的波动率倒数加权。全部为本仓库原创实现，仅用 numpy/pandas，离线且确定性。

数据约束与工程要点：
1. 本仓库只提供**收盘价面板**（无 high/low），因此 ATR 采用收盘价近似：
   令 high ≈ low ≈ close，则真实波幅 TR = max(|H-L|, |H-C_prev|, |L-C_prev|)
   退化为 |close_t - close_{t-1}|（即「绝对价格变动」），ATR 为其 Wilder 平滑。
   这是标准 TR 的下界近似，用于自适应通道宽度足够，但会系统性低估含日内振幅的波动。
2. 防未来函数：所有波动率/分位阈值/ATR 均为**滚动历史**统计；突破信号把通道
   上轨再 `shift(1)`（用昨日通道判断今日收盘）；regime 阈值用 `shift(1)` 的历史序列。
   即第 t 期权重只依赖截至 t（含 t 收盘）的信息，引擎还会再滞后一期。
3. 权重恒 >= 0，且逐行和 <= 1（不加杠杆），`_finalize` 做最后一道裁剪与行内缩放兜底。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-12


# --------------------------------------------------------------------------
# 模块内小工具（不从其它渠道 import 私有函数）
# --------------------------------------------------------------------------
def _market_returns(data: MarketData) -> pd.Series:
    """等权（每日再平衡）组合收益率序列，作为「市场」代理。"""
    return data.returns(1).mean(axis=1)


def _realized_vol(ret: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    """已实现波动率（年化）：滚动标准差 × sqrt(ppy)，min_periods=window（不足则 NaN）。"""
    return ret.rolling(window, min_periods=window).std() * np.sqrt(float(periods_per_year))


def _close_atr(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """收盘价近似 ATR（Wilder 平滑），逐列复用 `indicators.atr`。

    以 high = low = close 传入 `ind.atr`，其 `true_range` 退化为 |close.diff()|，
    因此结果等价于「平均绝对价格变动」。`ind.atr` 对 DataFrame 输入返回 ndarray，
    故这里逐列（Series）调用以保持 pandas 的 index/columns 语义。
    """
    cols = list(close.columns)
    parts: Dict[str, pd.Series] = {}
    for c in cols:
        s = close[c]
        parts[c] = ind.atr(s, s, s, window)
    return pd.concat(parts, axis=1).reindex(index=close.index, columns=cols)


def _state_machine(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """状态机持有：enter 置 1 并保持，exit_ 清 0，其间沿用上一状态（滞后带效果）。

    进入/退出使用不同阈值可显著降低日频 whipsaw 换手。
    """
    e = enter.reindex(index=exit_.index, columns=exit_.columns).fillna(False).values.astype(bool)
    x = exit_.fillna(False).values.astype(bool)
    out = np.zeros(e.shape, dtype=float)
    hold = np.zeros(e.shape[1], dtype=float)
    for t in range(e.shape[0]):
        row_e, row_x = e[t], x[t]
        hold = np.where(row_x, 0.0, np.where(row_e, 1.0, hold))
        out[t] = hold
    return pd.DataFrame(out, index=exit_.index, columns=exit_.columns)


def _state_machine_series(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """单序列版状态机（用于组合层面的 regime 开关）。"""
    e = enter.fillna(False).values.astype(bool)
    x = exit_.reindex(index=enter.index).fillna(False).values.astype(bool)
    out = np.zeros(e.shape[0], dtype=float)
    hold = 0.0
    for t in range(e.shape[0]):
        if x[t]:
            hold = 0.0
        elif e[t]:
            hold = 1.0
        out[t] = hold
    return pd.Series(out, index=enter.index)


def _equal_budget(state: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """0/1 持有状态 -> 等预算权重（每资产 1/N，组合最大满仓）。"""
    return state.fillna(0.0) / max(int(data.n_assets), 1)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、去 NaN、裁到 [0, cap]，并把行和超过 cap 的行按比例缩回。"""
    out = w.reindex(index=data.dates, columns=data.symbols)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=cap)
    total = out.sum(axis=1)
    factor = (cap / total.replace(0.0, np.nan)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


def _broadcast(scale: pd.Series, data: MarketData) -> pd.DataFrame:
    """把组合层面的敞口标量序列广播成等权权重面板（每资产 scale/N）。"""
    per = scale.reindex(data.dates).fillna(0.0) / max(int(data.n_assets), 1)
    return pd.DataFrame(
        np.repeat(per.values[:, None], max(int(data.n_assets), 1), axis=1),
        index=data.dates, columns=data.symbols,
    )


# --------------------------------------------------------------------------
# 策略 1：波动率目标（总敞口缩放 overlay）
# --------------------------------------------------------------------------
class VolTargetOverlay(Strategy):
    """波动率目标：等权组合按「目标波动 / 已实现波动」缩放总敞口，上限 1（不加杠杆）。"""

    name = "vol_target"
    channel = "volatility"
    universe = "overlay"
    long_only = True
    description = "对等权组合做波动率目标：按 目标波动/已实现波动 缩放总敞口，高波降仓、低波满仓，缩放上限 1（不加杠杆）。"
    hypothesis = (
        "核心假设：波动率具有强聚集性与可预测性，且高波动期单位风险收益更差（杠杆效应、"
        "被迫去杠杆与流动性收缩）。把组合波动锁定在目标水平可显著降低尾部回撤、改善 Sharpe；"
        "当目标波动高于市场已实现波动时策略退化为等权持有（不加杠杆），因此牛市低波阶段不占优；"
        "在波动骤升后快速回落的 V 型反转中会因滞后而踏空反弹。"
    )
    source = "风险管理范式——volatility targeting / 风险预算的总敞口缩放（本仓库原创实现）。"
    params = {
        "window": 20,          # 已实现波动窗口（日）
        "target_vol": 0.12,    # 年化目标波动
        "max_scale": 1.0,      # 缩放上限：1 表示不加杠杆
        "min_scale": 0.0,      # 缩放下限：0 表示允许完全空仓
        "vol_floor": 0.01,     # 已实现波动下限，避免除以近零波动导致敞口爆表
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        rv = _realized_vol(_market_returns(data), int(p["window"]), data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        scale = (float(p["target_vol"]) / rv).clip(lower=float(p["min_scale"]),
                                                   upper=float(p["max_scale"]))
        return _finalize(_broadcast(scale, data), data)


# --------------------------------------------------------------------------
# 策略 2：ATR 通道突破（逐资产择时）
# --------------------------------------------------------------------------
class AtrBreakout(Strategy):
    """ATR 通道突破：收盘上穿「均线 + k×ATR」做多，跌回均线（或下轨）离场，状态机持有。"""

    name = "atr_breakout"
    channel = "volatility"
    universe = "timing"
    long_only = True
    description = "ATR 通道突破：收盘价上穿「均线 + k×ATR」做多，跌回均线（或下轨）离场；状态机持有 + 通道滞后一期以降低换手。"
    hypothesis = (
        "核心假设：以波动自适应的通道宽度能区分「真突破」与「噪声」——波动放大时门槛自动抬高，"
        "波动收敛后的突破更可能延续为趋势（波动收缩形态 VCP 的思路）。"
        "在趋势/突破行情中盈利，在无方向的宽幅震荡市中会被反复打脸（连续小亏，靠偶发大趋势回本）；"
        "由于用收盘价近似 TR（无日内高低点），通道偏窄，触发频率高于真实 ATR 通道。"
    )
    source = "经典技术分析——ATR 通道 / 肯特纳通道 (Keltner channel) 突破范式（本仓库原创实现）。"
    params = {
        "ma_window": 20,        # 中轨均线窗口
        "atr_window": 14,       # ATR（Wilder）窗口
        "k": 2.0,               # 通道宽度倍数：上轨 = 均线 + k×ATR
        "atr_floor_pct": 0.002, # ATR 下限 = 0.2%×价格，避免完全无波动时通道宽度为 0
        "exit_band": 0.0,       # 离场缓冲（占中轨/下轨比例），>0 可进一步降低换手
        "exit_ref": "mid",      # 离场参考线："mid"（均线）或 "lower"（下轨）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        mid = ind.sma(close, int(p["ma_window"]))
        atr = _close_atr(close, int(p["atr_window"]))
        # ATR 下限：完全平坦的价格序列 |diff|=0 会让通道退化为均线本身
        atr = np.maximum(atr, float(p["atr_floor_pct"]) * close)
        k = float(p["k"])
        band = float(p["exit_band"])
        upper_prev = (mid + k * atr).shift(1)          # 用昨日通道判断今日收盘
        lower_prev = (mid - k * atr).shift(1)
        mid_prev = mid.shift(1)
        valid = mid.notna() & atr.notna() & upper_prev.notna()
        enter = (close > upper_prev) & valid
        if str(p["exit_ref"]) == "lower":
            exit_ = (close < lower_prev * (1.0 - band)) & valid & lower_prev.notna()
        else:
            exit_ = (close < mid_prev * (1.0 - band)) & valid & mid_prev.notna()
        return _finalize(_equal_budget(_state_machine(enter, exit_), data), data)


# --------------------------------------------------------------------------
# 策略 3：波动率状态过滤（risk-off 开关）
# --------------------------------------------------------------------------
class VolRegimeFilter(Strategy):
    """波动率状态过滤：市场已实现波动低于其滚动分位阈值时持有等权组合，否则空仓。"""

    name = "vol_regime_filter"
    channel = "volatility"
    universe = "overlay"
    long_only = True
    description = "波动率 regime 过滤：等权组合的已实现波动低于其滚动分位阈值（默认 70% 分位）时满仓等权，突破极端高波分位（默认 90%）时空仓（risk-off），双分位滞后带减少状态抖动。"
    hypothesis = (
        "核心假设：波动率状态可预测（聚集性）且高波动状态下的收益分布显著更差、更偏负"
        "（低波动异象 + 波动率与收益的负相关）。用**滚动分位**而非绝对阈值可自适应不同资产/时代的"
        "波动水平；双分位（进/出）滞后带避免临界抖动。"
        "失效场景：低波之后的突发崩盘（分位阈值滞后于跳升的波动）、以及高波动伴随强反弹的市场"
        "（如政策底后的急涨）会踏空；lookback 越长越稳但反应越慢。"
    )
    source = "波动率 regime switching / risk-off 择时过滤范式（本仓库原创实现）。"
    params = {
        "vol_window": 20,     # 已实现波动窗口
        "lookback": 250,      # 分位阈值的滚动回看窗口
        "q_enter": 0.7,       # 低于该分位 -> 进入 risk-on（波动不算极端即可持有）
        "q_exit": 0.9,        # 高于该分位 -> 退出（极端高波 risk-off），与 q_enter 形成滞后带
        "vol_floor": 0.01,    # 波动下限，避免零波动导致分位阈值退化
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        rv = _realized_vol(_market_returns(data), int(p["vol_window"]), data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        hist = rv.shift(1)                             # 阈值只用截至 t-1 的历史
        lb = int(p["lookback"])
        min_obs = max(2, lb // 2)
        thr_enter = hist.rolling(lb, min_periods=min_obs).quantile(float(p["q_enter"]))
        thr_exit = hist.rolling(lb, min_periods=min_obs).quantile(float(p["q_exit"]))
        ready = rv.notna() & thr_enter.notna() & thr_exit.notna()
        enter = (rv < thr_enter) & ready
        exit_ = (rv > thr_exit) & ready
        on = _state_machine_series(enter, exit_)
        return _finalize(_broadcast(on, data), data)


# --------------------------------------------------------------------------
# 策略 4：波动率倒数加权的时序动量
# --------------------------------------------------------------------------
class VolScaledMomentum(Strategy):
    """时序动量 + 波动率倒数加权：只持有动量为正的资产，波动越大仓位越小。"""

    name = "vol_scaled_momentum"
    channel = "volatility"
    universe = "timing"
    long_only = True
    description = "时序动量（跳过最近数期）为正的资产才持有，权重按已实现波动倒数分配（波动越大仓位越小），总敞口 = 正动量资产占比。"
    hypothesis = (
        "核心假设：动量信号的风险调整强度与波动成反比——同样的涨幅，低波动资产更可能来自"
        "持续性资金流而非噪声，因此按 1/vol 加权近似最大化组合 Sharpe（等风险贡献的简化版）；"
        "总敞口用「正动量资产占比」(breadth) 决定，等价于在市场普跌时自动降仓（绝对动量过滤）。"
        "失效场景：低波动资产同时缺乏弹性时会跑输等权；高波动急跌后的反弹被系统性低配；"
        "breadth 项在震荡市中会频繁改变总敞口，增加换手。"
    )
    source = "时序动量 (time-series momentum) + 波动率缩放 (inverse-vol weighting) 的组合范式（本仓库原创实现）。"
    params = {
        "mom_window": 60,     # 动量回看窗口
        "skip": 5,            # 跳过最近 N 期，规避短期反转污染
        "vol_window": 20,     # 已实现波动窗口
        "vol_floor": 0.05,    # 年化波动下限，避免近零波动资产吃掉全部预算
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        mom = close.shift(int(p["skip"])).pct_change(int(p["mom_window"]))   # 仅用历史价格
        rv = ind.realized_vol(close, int(p["vol_window"]), data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        signal = (mom > 0.0).where(mom.notna(), False).astype(float)
        raw = signal / rv                                # 波动率倒数加权
        total = raw.sum(axis=1)
        share = raw.div(total.where(total > _EPS), axis=0).fillna(0.0)   # 行内归一 -> 和为 1
        breadth = (signal.sum(axis=1) / max(int(data.n_assets), 1)).clip(upper=1.0)
        w = share.mul(breadth, axis=0).fillna(0.0)       # 行和 = breadth <= 1
        return _finalize(w, data)
