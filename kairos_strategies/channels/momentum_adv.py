"""渠道：momentum_adv —— 学术界更「进阶」的动量变体（针对动量因子的已知缺陷）。

收集途径与范式（**全部为本仓库原创实现**，只用 numpy/pandas；因子、截面排序、
多空配权、分组等辅助函数均在本模块内自实现，不 import 其它渠道的私有函数）：

普通动量（momentum 渠道的 ts/xs/52w/dual、long_short 渠道的 xs_momentum_ls /
residual_momentum_ls）只回答「谁涨得多就买谁」，但学术研究发现原始动量存在几个
系统性缺陷，本渠道的四个变体分别对症下药：

  1. ``frog_in_pan``            路径依赖 / 信息离散度：同样涨幅，由许多小涨日平滑
                                累积（"温水煮青蛙"）的动量比单日跳空式动量更可持续；
                                做多「平滑上涨」、规避「跳空动量」（只做多，行和≈1）。
  2. ``vol_managed_momentum``   动量崩溃 (momentum crash)：先构造截面动量组合（做多
                                强者），再按**市场层**已实现波动缩放总敞口（高波降杠杆），
                                削薄危机后 V 型反弹里的剧烈回撤（只做多，行和≤1）。
  3. ``momentum_spread_timing`` 崩溃 + 拥挤：做多赢家/做空输家（WML，美元中性），但仅当
                                「动量价差动量」为正（赢家组仍相对输家组走强）时才持仓，
                                价差转弱即整行平仓避险（可多空，行和≈0、绝对值和≤1）。
  4. ``group_momentum``         个股噪声 / 特质反转 / 拥挤：把资产按索引等分为若干组，
                                以组内成员动量均值给组打分，做多最强组、做空最弱组
                                （组内等权），把下注单元从个股抬升到群组（可多空，行和≈0）。

防未来函数：所有动量/波动/价差/分组统计在第 t 期只使用截至 t（含 t 收盘）的价格，
且均以 shift（历史价）或滚动**尾部窗口**构造；截面排序只用当期截面信息。引擎还会再
滞后一期，双重保险。排序并列按列索引稳定打破（rank method='first' / np.lexsort），
同一数据多次调用逐位一致（确定性），篡改 t 之后的价格不改变 t 及之前任何一行权重。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-9       # 排名 / 取整容差
_TINY = 1e-12     # 除零 / 退化保护阈值


# ------------------------------------------------------------------ 通用工具（本模块自实现）

def _prices(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64、裁到正数（保证比值/对数安全），对齐 index/columns。"""
    return data.prices.astype("float64").clip(lower=1e-9)


def _returns(data: MarketData) -> pd.DataFrame:
    """日度简单收益面板；首期无昨收置 0（非未来信息），其余 NaN 一律填 0。"""
    p = _prices(data)
    r = p.pct_change()
    if len(r) > 0:
        r.iloc[0] = 0.0
    return r.fillna(0.0)


def _market_returns(data: MarketData) -> pd.Series:
    """等权（每日再平衡）组合收益序列，作为「市场层」代理。"""
    return _returns(data).mean(axis=1)


def _realized_vol(ret: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    """尾部窗口年化已实现波动率（只用历史，min_periods=window，预热期为 NaN）。"""
    w = max(int(window), 2)
    return ret.rolling(w, min_periods=w).std() * np.sqrt(float(periods_per_year))


def _xs_momentum_score(data: MarketData, lookback: int, skip: int) -> pd.DataFrame:
    """截面动量分数：从 t-lookback 到 t-skip 的累计收益（只用 shift 历史价，绝不偷看未来）。"""
    p = _prices(data)
    lb = max(int(lookback), 2)
    sk = max(int(skip), 0)
    return p.shift(sk) / p.shift(lb) - 1.0


def _top_k_mask(factor: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    """按 factor 降序，每行选中前 ``ceil(n_valid*top_frac)`` 个（有有效值时至少 1 个）。

    返回布尔 DataFrame；并列按列顺序（``method='first'``）确定性打破，NaN 一律不选。
    """
    valid = factor.notna()
    n_valid = valid.sum(axis=1).to_numpy(dtype=float)                       # (T,)
    rank_desc = factor.rank(axis=1, ascending=False, method="first").to_numpy()  # (T, n)
    # 减极小量再上取整，避免 n_valid*top_frac 恰为整数时的浮点上溢
    k = np.ceil(n_valid * float(top_frac) - _EPS)
    k = np.where(n_valid > 0, np.maximum(k, 1.0), 0.0)                       # (T,)
    mask = (rank_desc <= k[:, None]) & valid.to_numpy()
    return pd.DataFrame(mask, index=factor.index, columns=factor.columns)


def _normalize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    """被选中资产等权归一：每行和为 1（该行无选中时保持全 0）。"""
    w = selected.astype(float)
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(row_sum, axis=0).fillna(0.0)


def _quantile_long_short(scores: pd.DataFrame, data: MarketData, top_frac: float,
                         leg_budget: float) -> pd.DataFrame:
    """把「越高越好」的截面分数转成美元中性、无杠杆的多空目标权重面板。

    逐行（只用当期截面分数）：有效分数降序，前 ``k`` 名做多、后 ``k`` 名做空，
    ``k = clip(ceil(top_frac*n_valid), 1, n_valid//2)`` 保证两腿非空且不重叠；并列按
    列索引升序稳定打破（``np.lexsort``）。两腿各自等权归一到 ``leg_budget``，故每行
    权重和 ≈ 0（多空预算精确抵消）、绝对值之和 = 2*leg_budget ≤ 1。有效分数 < 2 个
    （凑不出两腿）或预热期 -> 整行空仓。输出对齐 ``data``，不含 NaN。
    """
    aligned = (scores.reindex(index=data.dates, columns=data.symbols)
                     .apply(pd.to_numeric, errors="coerce"))
    v = aligned.to_numpy(dtype="float64")
    v[~np.isfinite(v)] = np.nan
    T, n = v.shape
    out = np.zeros((T, n), dtype="float64")
    frac = float(top_frac)
    budget = float(leg_budget)
    for t in range(T):
        row = v[t]
        idx = np.flatnonzero(np.isfinite(row))
        nv = int(idx.size)
        if nv < 2:
            continue                                            # 凑不出两腿 -> 空仓
        k = int(np.ceil(nv * frac - _EPS))                      # top/bottom 分位名额
        k = max(1, min(k, nv // 2))                             # 两腿非空且不重叠
        vals = row[idx]
        order = np.lexsort((idx, -vals))                        # 主键分数降序，并列按列索引
        out[t, idx[order[:k]]] = budget / k                     # 多头腿 +budget/k
        out[t, idx[order[nv - k:]]] = -budget / k               # 空头腿 -budget/k
    return pd.DataFrame(out, index=data.dates, columns=data.symbols)


def _assign_groups(n: int, n_groups: int):
    """按列索引把 ``n`` 个资产等分为 ``n_groups`` 个**连续块**（确定性、与数据无关）。

    返回 list[ndarray]（每块是列位置索引），或 ``None``（资产/组数不足以分出两组）。
    """
    n = int(n)
    g = int(n_groups)
    if n < 2 or g < 2:
        return None
    g = min(g, n)                                               # 每组至少 1 个成员
    if g < 2:
        return None
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(np.arange(n), g)]


def _finalize_long(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """只做多收尾：对齐 index/columns、去 NaN、裁到 [0, cap]，行和超过 cap 的行按比例缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0).clip(lower=0.0, upper=cap))
    total = out.sum(axis=1)
    factor = (cap / total.replace(0.0, np.nan)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


def _finalize_neutral(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """多空收尾：对齐、去 NaN、裁到 [-cap, cap]，绝对值和超过 cap 的行按比例缩回（保持行和≈0）。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0).clip(lower=-cap, upper=cap))
    gross = out.abs().sum(axis=1)
    factor = (cap / gross.replace(0.0, np.nan)).where(gross > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


# ------------------------------------------------------------------ 策略 1：温水煮青蛙（路径信息离散度）

class FrogInPanStrategy(Strategy):
    name = "frog_in_pan"
    channel = "momentum_adv"
    universe = "cross_section"
    long_only = True
    description = ("温水煮青蛙动量：用过去 N 期「上涨日占比」与收益路径的符号一致性(|Σr|/Σ|r|)"
                   "度量信息离散度，只做多由许多小涨日平滑累积上涨的资产、规避单日跳空式动量，"
                   "被选中者截面等权归一（每行和≈1）。")
    hypothesis = ("针对动量的『路径依赖 / 信息到达方式』缺陷：同样的累计涨幅，若由持续、离散度低的"
                  "小幅上涨累积而成（青蛙效应），说明信息在逐步扩散、动量更可信也更可持续；若涨幅主要"
                  "由少数跳空/暴涨贡献（信息离散度高），多为一次性事件驱动、随后极易反转。故做多『平滑"
                  "上涨』、过滤『跳空动量』能提升动量的稳健性并降低崩溃概率。失效条件：平滑趋势的末端"
                  "拐点（历史平滑度无法预知反转）、普跌或普涨到无平滑标的可选时信号退化，短窗口下"
                  "上涨日占比对微观噪声敏感、换手偏高。")
    source = ("进阶动量变体——Frog-in-the-Pan 信息离散度（Da-Liu-Schaumburg 思路，本仓库原创 "
              "numpy/pandas 实现）；与 momentum 渠道的 ts/xs/high_52w/dual（只看涨幅大小）、"
              "long_short 渠道的 xs_momentum_ls（多空两端下注）在信号构造上均不同，此处专挑『涨得平滑』。")
    params = {"window": 60,             # 路径信息离散度观察窗口（约一季度）
              "top_frac": 1.0 / 3.0,    # 截面做多前 1/3 分位
              "min_up_ratio": 0.5}      # 上涨日占比门槛：过滤跳空/震荡，只留平滑上涨

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _prices(data)
        r = _returns(data)
        n = max(int(self.params["window"]), 2)
        # 方向：过去 n 期累计收益（只用历史价）
        net = p / p.shift(n) - 1.0
        # 上涨日占比：尾部窗口内 r>0 的比例（青蛙效应的核心度量）
        up_ratio = (r > 0).astype(float).rolling(n, min_periods=n).mean()
        # 符号一致性 / 路径效率：|Σr| / Σ|r| ∈ [0,1]，越高路径越「直」越平滑
        sum_r = r.rolling(n, min_periods=n).sum()
        sum_abs = r.abs().rolling(n, min_periods=n).sum()
        eff = (sum_r.abs() / sum_abs.where(sum_abs > _TINY)).fillna(0.0)
        # frog 分：平滑且持续上涨者高（跳空上涨 up_ratio 低 -> 分低）
        frog = up_ratio * eff
        # 门槛：动量为正 + 上涨日占比达标，才够格被做多（规避跳空式动量）
        gate = (net > 0.0) & up_ratio.notna() & (up_ratio >= float(self.params["min_up_ratio"]))
        score = frog.where(gate)                                # 非门槛 -> NaN -> 不选
        selected = _top_k_mask(score, self.params["top_frac"])
        return _finalize_long(_normalize_selected(selected), data)


# ------------------------------------------------------------------ 策略 2：波动管理动量（控制动量崩溃）

class VolManagedMomentumStrategy(Strategy):
    name = "vol_managed_momentum"
    channel = "momentum_adv"
    universe = "cross_section"
    long_only = True
    description = ("波动管理动量：先按截面动量做多强者构成动量组合（每行和≈1），再按**市场层**已实现"
                   "波动对总敞口整体缩放（高波降杠杆、上限满仓不加杠杆），以控制动量崩溃；只做多、每行和≤1。")
    hypothesis = ("针对动量的『崩溃 (momentum crash)』缺陷：截面动量在低波平稳期表现最好，而在高波/"
                  "危机后的 V 型反弹期，因前期最弱者弹性最大而多头跑输、空头（此处不做空但组合整体）"
                  "剧烈回撤。用市场已实现波动对动量组合整体缩放（高波降仓、低波满仓）能显著削薄尾部、"
                  "提升夏普，且不加杠杆。失效条件：波动骤升后又快速回落时缩放滞后、踏空反弹；低波牛市里"
                  "因缩放上限为 1 退化为满仓动量、相对无管理版没有超额；市场层波动是个股波动的粗糙代理。")
    source = ("进阶动量变体——波动管理动量 (volatility-managed momentum, Barroso-Santa-Clara / "
              "Moreira-Muir 思路，本仓库原创实现)；与 volatility 渠道的 vol_target（缩放对象是**等权"
              "组合**、名字不同）区别在于：此处缩放的是**截面动量组合**，目标是压制动量崩溃而非通用风险预算。")
    params = {"lookback": 126,          # 截面动量观察窗口（约半年）
              "skip": 21,               # 剔除最近约 1 个月，规避短期反转污染
              "top_frac": 1.0 / 3.0,    # 做多前 1/3 分位强者
              "vol_window": 21,         # 市场层已实现波动窗口（约一个月）
              "target_vol": 0.15,       # 年化目标波动：scale = target / realized
              "max_scale": 1.0,         # 缩放上限：1 表示不加杠杆
              "vol_floor": 0.01}        # 已实现波动下限，避免近零波动导致敞口爆表

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        prm = self.params
        score = _xs_momentum_score(data, int(prm["lookback"]), int(prm["skip"]))
        sleeve = _normalize_selected(_top_k_mask(score, prm["top_frac"]))   # 动量组合，行和≈1
        rv = _realized_vol(_market_returns(data), int(prm["vol_window"]),
                           data.periods_per_year).clip(lower=float(prm["vol_floor"]))
        scale = (float(prm["target_vol"]) / rv).clip(lower=0.0, upper=float(prm["max_scale"]))
        scale = scale.fillna(0.0)                                           # 预热期 -> 空仓
        w = sleeve.mul(scale, axis=0)                                       # 行和 = scale ≤ 1
        return _finalize_long(w, data)


# ------------------------------------------------------------------ 策略 3：多空动量价差择时

class MomentumSpreadTimingStrategy(Strategy):
    name = "momentum_spread_timing"
    channel = "momentum_adv"
    universe = "cross_section"
    long_only = False
    description = ("多空动量价差择时：做多赢家/做空输家构成美元中性 WML（行和≈0、绝对值和≤1），但仅当"
                   "『动量价差动量』为正（赢家组近期仍相对输家组走强）时才持仓，价差转弱即整行平仓避险。")
    hypothesis = ("针对动量的『崩溃 + 拥挤』缺陷：多空动量价差 (WML) 在价差持续走阔时最稳，而在价差见顶"
                  "收敛时最易崩溃——拥挤的同向资金一旦反向解仓、或危机后前期弱者暴力反弹，WML 会短期内"
                  "巨亏。用价差组合自身的近期已实现收益（赢家腿减输家腿）作为『价差动量』开关，在其转负时"
                  "空仓，可规避大部分崩溃段落。失效条件：价差反复拉锯的震荡期会频繁开关、放大换手与成本，"
                  "并被假信号两头打脸；开关本身滞后，对突发单日崩溃反应不及。")
    source = ("进阶动量变体——动量价差/波动择时 (momentum spread timing, Daniel-Moskowitz 动量崩溃"
              "择时思路，本仓库原创实现)；与 long_short 渠道的 xs_momentum_ls（**始终满仓**多空）不同，"
              "此处按价差动量**动态开关**多空敞口，崩溃来临前主动离场。")
    params = {"lookback": 126,          # 截面动量观察窗口
              "skip": 21,               # 剔除最近约 1 个月
              "top_frac": 1.0 / 3.0,    # 多空两腿各取分位
              "leg_budget": 0.5,        # 单腿预算：多头 +0.5、空头 -0.5，行和≈0
              "spread_window": 21}      # 「价差动量」= 价差组合收益的近期滚动均值窗口

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        prm = self.params
        score = _xs_momentum_score(data, int(prm["lookback"]), int(prm["skip"]))
        wml = _quantile_long_short(score, data, prm["top_frac"], prm["leg_budget"])  # 行和≈0
        r = _returns(data)
        # 价差组合的已实现日收益 = Σ_i wml_i·r_i（赢家腿减输家腿，符号即价差方向）
        spread_ret = wml.mul(r).sum(axis=1)
        m = max(int(prm["spread_window"]), 2)
        spread_mom = spread_ret.rolling(m, min_periods=m).mean()   # 价差动量（尾部窗口）
        gate = (spread_mom > 0.0).astype(float)                    # 仅价差走强时开仓
        w = wml.mul(gate, axis=0)                                  # 行和仍≈0，绝对值和≤1
        return _finalize_neutral(w, data)


# ------------------------------------------------------------------ 策略 4：群组动量

class GroupMomentumStrategy(Strategy):
    name = "group_momentum"
    channel = "momentum_adv"
    universe = "cross_section"
    long_only = False
    description = ("群组动量：按列索引把资产等分为若干连续组，逐日以组内成员截面动量均值给每个组打分，"
                   "做多最强组、做空最弱组（组内等权），美元中性（行和≈0、绝对值和≤1）。")
    hypothesis = ("针对个股动量的『噪声 / 特质反转 / 拥挤』缺陷：单只资产的动量易被一次性事件与微观噪声"
                  "污染，也更容易被拥挤资金抢跑；把动量先聚合到组（板块/群组）层面再排序，能分散掉特质噪声、"
                  "捕捉更稳健的群组相对强弱，做多最强组、做空最弱组，价差来自群组间而非个股间的基本面/资金"
                  "分化。失效条件：组间高度同步（相关性骤升、离散度消失）时无强弱可分；固定的索引分组规则若"
                  "与真实市场结构（行业/风格）错配，会把异质资产混进同组而稀释信号；组数过少则颗粒太粗。")
    source = ("进阶动量变体——群组/行业动量 (group & industry momentum, Moskowitz-Grinblatt 行业动量"
              "思路，本仓库原创实现，用**索引等分**做确定性、可复现、无未来信息的分组)；与个股层面的 "
              "xs_momentum_ls（以单只资产为下注单元）不同，此处以**组**为下注单元、组内等权。")
    params = {"lookback": 126,          # 组内成员截面动量窗口
              "skip": 21,               # 剔除最近约 1 个月
              "n_groups": 3,            # 组数（按列索引等分为连续块）
              "leg_budget": 0.5}        # 单腿预算：最强组 +0.5、最弱组 -0.5，行和≈0

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        prm = self.params
        score = _xs_momentum_score(data, int(prm["lookback"]), int(prm["skip"]))
        aligned = (score.reindex(index=data.dates, columns=data.symbols)
                        .apply(pd.to_numeric, errors="coerce"))
        v = aligned.to_numpy(dtype="float64")
        v[~np.isfinite(v)] = np.nan
        T, n = v.shape
        groups = _assign_groups(n, int(prm["n_groups"]))
        out = np.zeros((T, n), dtype="float64")
        budget = float(prm["leg_budget"])
        if groups is not None:
            for t in range(T):
                row = v[t]
                gmean = np.full(len(groups), np.nan, dtype="float64")
                for gi, members in enumerate(groups):
                    vals = row[members]
                    fin = vals[np.isfinite(vals)]
                    if fin.size:
                        gmean[gi] = fin.mean()                  # 组分数 = 组内成员动量均值
                valid = np.flatnonzero(np.isfinite(gmean))
                if valid.size < 2:
                    continue                                    # 凑不出「最强/最弱」两组 -> 空仓
                gv = gmean[valid]
                order = np.lexsort((valid, -gv))                # 组分数降序，并列按组索引
                strongest = groups[valid[order[0]]]
                weakest = groups[valid[order[-1]]]
                s_mem = strongest[np.isfinite(row[strongest])]  # 仅给有信息的成员配权
                w_mem = weakest[np.isfinite(row[weakest])]
                if s_mem.size == 0 or w_mem.size == 0:
                    continue
                out[t, s_mem] = budget / s_mem.size             # 最强组等权做多
                out[t, w_mem] = -budget / w_mem.size            # 最弱组等权做空
        w = pd.DataFrame(out, index=data.dates, columns=data.symbols)
        return _finalize_neutral(w, data)
