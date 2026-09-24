"""渠道：defensive —— 防御类策略（低 beta / 防御质量 / 回撤规避）。

主题：不预测「谁会涨」，而是系统性地**回避高风险暴露**——把资金从「对市场敏感、
波动大、正处于深回撤」的资产挪向「低 beta、低波动、贴近自身高点、慢均线之上」的
资产，靠**风险端**而非收益端赚长期风险调整后收益。四个策略全部为本仓库 100% 原创
实现，只用 numpy/pandas，离线、确定性，辅助函数（beta 估计与收缩、water-filling
上限再分配、排名配权、多空腿归一）均在本模块内自实现，不 import 其它渠道私有函数。

与既有渠道的分工（避免重复/重名）：
  * ``factor.low_volatility`` / ``long_short.lowvol_ls``：信号是**波动率单因子**
    （只做多 / 分位多空）。本渠道的核心解释变量是**相对等权市场的 beta**
    （``betting_against_beta``、``low_beta_timing``）与**回撤+趋势的复合质量**
    （``defensive_quality``、``drawdown_averse``），不是波动率单因子排序。
  * ``regime.beta_timing``：以 1/N 等权预算为基座、用乘数 ``g=clip(1+k(1-β))``
    做「组合内部风险再分配」（行和≈1 来自等权 beta 均值恒为 1 的恒等式）。本渠道
    ``low_beta_timing`` 是**逆 beta 直接配权**：``raw ∝ 1/β_shrunk``、逐行归一到
    和=1、beta 先做朝当期截面均值的收缩（Vasicek/Blume 思路的固定强度简化）、再对
    单资产权重上限做 water-filling 再分配；``betting_against_beta`` 更是本渠道唯一
    的多空策略（美元中性、行和≈0），构造与命名均不同。
  * ``allocation.inverse_vol`` / ``risk_parity_alloc``：按**协方差/波动**分配预算
    （风险预算视角）。本渠道按 **beta 与回撤状态**分配（防御异象视角），且带 beta
    收缩、单资产上限与回撤门限。
  * ``momentum.high_52w``：把「接近 52 周新高」当**动量突破**信号做多。本渠道
    ``drawdown_averse`` 把「远离滚动高点 = 深回撤」当**风险规避**信号做减仓/剔除，
    并叠加低波倾斜与组合层健康度缩放（行和可 < 1，差额是现金）。
  * ``riskmgmt.drawdown_throttle`` / ``cppi`` / ``circuit_breaker``：都是**组合层
    overlay**（对等权基准净值整体降杠杆）。``drawdown_averse`` 是**逐资产**回撤
    闸门 + 截面配权，粒度与信号来源均不同。

防未来函数：第 t 期权重只用「截至 t（含 t 收盘）」的价格/收益——beta、波动、
滚动高点/回撤、慢均线全部是 trailing 窗口统计（``min_periods=window``），截面排名/
z-score/中位数只用**当期截面**，不存在任何全样本统计量；预热期（信息不足）按策略
分别回退到「等权满仓」（``low_beta_timing``）或「空仓」（其余三个），绝不用未来
数据回填。引擎还会再滞后一期，双重保险。

权重界限：
  * ``betting_against_beta``：long_only=False，逐行权重和 ≈ 0（美元中性），
    每行绝对值之和 = 2 × leg_budget ≤ 1（不加杠杆），单资产 ∈ [-1, 1]；
  * ``low_beta_timing`` / ``defensive_quality``：long_only=True，权重 ≥ 0，
    逐行和 ≈ 1（满仓配置，无杠杆）；
  * ``drawdown_averse``：long_only=True，权重 ≥ 0，逐行和 ≤ 1（差额为现金）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-12      # 除零 / 归一保护
_TINY = 1e-16     # 市场方差下限：低于该值视为 beta 未知
_PRICE_FLOOR = 1e-9


# --------------------------------------------------------------------------
# 面板预处理（本模块自实现，不从其它渠道 import 私有函数）
# --------------------------------------------------------------------------
def _panel(data: MarketData) -> pd.DataFrame:
    """价格面板：对齐 index/columns、转 float64、裁到正数（保证比值安全）。"""
    p = (data.prices.reindex(index=data.dates, columns=data.symbols)
         .apply(pd.to_numeric, errors="coerce")
         .astype("float64"))
    return p.clip(lower=_PRICE_FLOOR)          # NaN 原样保留（clip 不填充）


def _returns(p: pd.DataFrame) -> pd.DataFrame:
    """日度简单收益：首期无昨收置 0（非未来信息），其余 NaN 一律填 0。"""
    r = p.pct_change()
    if len(r) > 0:
        r.iloc[0] = 0.0
    return r.fillna(0.0)


def _market_returns(r: pd.DataFrame) -> pd.Series:
    """等权（每日再平衡）组合收益，作为「市场」代理（只用当期截面信息）。"""
    return r.mean(axis=1)


# --------------------------------------------------------------------------
# beta：滚动 OLS 估计 + 朝截面均值收缩
# --------------------------------------------------------------------------
def _rolling_beta(r: pd.DataFrame, mkt: pd.Series, window: int,
                  var_eps: float = _TINY) -> pd.DataFrame:
    """逐资产对等权市场的滚动 OLS beta（带截距，trailing 窗口，min_periods=window）。

    用「均值差」形式向量化：cov = E[x·y] − E[x]E[y]，var = E[y²] − E[y]²，
    beta = cov / var；市场方差 ≤ var_eps（退化）或样本不足 → NaN（预热期未知）。
    第 t 期只使用 r[t-window+1 .. t]，严格无未来。
    """
    w = max(int(window), 2)
    mean_x = r.rolling(w, min_periods=w).mean()
    mean_y = mkt.rolling(w, min_periods=w).mean()
    mean_xy = r.mul(mkt, axis=0).rolling(w, min_periods=w).mean()
    mean_yy = (mkt * mkt).rolling(w, min_periods=w).mean()
    cov = mean_xy.sub(mean_x.mul(mean_y, axis=0))
    var = (mean_yy - mean_y.pow(2)).clip(lower=0.0)
    return cov.div(var.where(var > float(var_eps)), axis=0)


def _shrink_to_cross_section(beta: pd.DataFrame, strength: float) -> pd.DataFrame:
    """把 beta 朝**当期截面均值**收缩：``β* = (1-s)·β + s·mean_t(β)``。

    Vasicek(1973)/Blume 式收缩的固定强度简化：样本 beta 含大量估计噪声（尤其窗口
    短、特质波动大时），朝截面均值收缩可显著降低「beta 排序被噪声主导」的概率。
    均值只用**当期截面**（逐行、忽略 NaN），不涉及任何未来信息；NaN 原样保留。
    """
    s = float(min(max(float(strength), 0.0), 1.0))
    if s <= 0.0:
        return beta
    mu = beta.mean(axis=1)                       # 逐行截面均值（skipna）
    return beta.mul(1.0 - s).add(mu.mul(s), axis=0)


def _estimated_beta(data: MarketData, window: int, shrink: float) -> pd.DataFrame:
    """「等权市场 beta」的统一入口：滚动 OLS -> 截面收缩（预热期为 NaN）。"""
    r = _returns(_panel(data))
    beta = _rolling_beta(r, _market_returns(r), window)
    return _shrink_to_cross_section(beta, shrink)


def _current_drawdown(p: pd.DataFrame, window: int) -> pd.DataFrame:
    """当期回撤深度：``dd = 1 − p_t / max(p_{t-window+1..t})`` ∈ [0, 1]。

    滚动高点为 trailing 最大值（含当期），故 dd ≥ 0；贴近自身滚动高点 → dd ≈ 0，
    深回撤 → dd → 1。样本不足（预热期）→ NaN。
    """
    w = max(int(window), 2)
    hi = p.rolling(w, min_periods=w).max()
    dd = 1.0 - p / hi.where(hi > _EPS)
    return dd.clip(lower=0.0, upper=1.0)


def _realized_vol(r: pd.DataFrame, window: int, periods_per_year: int) -> pd.DataFrame:
    """尾部窗口年化已实现波动（min_periods=window，预热期 NaN）。"""
    w = max(int(window), 2)
    return r.rolling(w, min_periods=w).std() * np.sqrt(float(periods_per_year))


# --------------------------------------------------------------------------
# 截面工具：行 z-score / 排名配权 / 上限 water-filling
# --------------------------------------------------------------------------
def _row_z(x: pd.DataFrame) -> pd.DataFrame:
    """截面标准化（逐行减均值、除标准差 ddof=1）；只用当期截面，无未来信息。

    某行有效值 < 2 或截面标准差 ≈ 0（整行退化）→ 该行全 NaN（当日不参与排序）。
    """
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > _EPS), axis=0)


def _cap_water_fill(w: np.ndarray, cap: float, iters: int) -> np.ndarray:
    """对「行和 = 1、非负」的权重矩阵按单资产上限 ``cap`` 做 water-filling 再分配。

    超出上限的部分被削平，并按剩余容量比例分给未触顶的资产，迭代至无人超限
    （最多 n 轮即收敛，``iters ≥ n+1`` 时精确）；若整行都已触顶无处可放，余量留在
    现金（行和 ≤ 1）。纯逐行代数运算，确定性、无未来信息。
    """
    out = np.array(w, dtype="float64", copy=True)
    tol = 1e-15
    for _ in range(max(int(iters), 1)):
        over = out > cap + tol
        if not bool(over.any()):
            break
        excess = np.where(over, out - cap, 0.0)
        out = np.where(over, cap, out)
        room = np.where(over, 0.0, np.maximum(cap - out, 0.0))
        tot_room = room.sum(axis=1, keepdims=True)
        tot_excess = excess.sum(axis=1, keepdims=True)
        ok = tot_room > 1e-18
        out = out + np.where(ok, tot_excess * room / np.where(ok, tot_room, 1.0), 0.0)
    return np.minimum(out, cap)


def _normalize_rows(raw: np.ndarray) -> np.ndarray:
    """逐行归一到和 = 1（全 0 / 全 NaN 行保持 0，即当日空仓）。"""
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    tot = raw.sum(axis=1, keepdims=True)
    ok = tot > _EPS
    return np.where(ok, raw / np.where(ok, tot, 1.0), 0.0)


def _rank_top_weights(scores: pd.DataFrame, data: MarketData,
                      top_frac: float) -> pd.DataFrame:
    """把「越高越好」的截面分数转成只做多、逐行和 = 1 的**排名线性**权重面板。

    逐行（只用当期截面分数）：
      1. 分数降序排名（并列按列索引先后稳定打破，``method='first'``），选中前
         ``k = ceil(top_frac × n_valid)`` 名；NaN 分数（预热期）一律不选；
      2. 权重 ∝ ``k + 1 − rank``（第 1 名 k、第 k 名 1）——**纯排名**线性倾斜，
         对分数离群值稳健，且单调；
      3. 逐行归一到和 = 1；无有效分数（k = 0）→ 整行空仓。
    """
    s = (scores.reindex(index=data.dates, columns=data.symbols)
         .apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan))
    sv = s.to_numpy(dtype="float64")
    valid = np.isfinite(sv)
    nv = valid.sum(axis=1).astype("float64")
    k = np.ceil(nv * float(top_frac) - 1e-9)
    k = np.clip(k, 1.0, np.maximum(nv, 0.0))          # 至少 1 名，至多全部
    k = np.where(nv >= 1.0, k, 0.0)                    # 无有效分数 -> 空仓
    rk = s.rank(axis=1, ascending=False, method="first",
                na_option="keep").to_numpy(dtype="float64")
    sel = valid & (rk <= k[:, None])
    raw = np.where(sel, k[:, None] + 1.0 - np.where(sel, rk, 0.0), 0.0)
    w = _normalize_rows(raw)
    return pd.DataFrame(w, index=data.dates, columns=data.symbols)


# --------------------------------------------------------------------------
# 权重装配
# --------------------------------------------------------------------------
def _finalize_long(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """只做多兜底：对齐 index/columns、NaN→0、裁到 [0, cap]，行和超 cap 者等比缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0).clip(lower=0.0, upper=float(cap)))
    total = out.sum(axis=1)
    over = total > float(cap) + 1e-15
    scale = pd.Series(1.0, index=out.index, dtype="float64")
    if bool(over.any()):
        scale[over] = float(cap) / total[over]
    return out.mul(scale, axis=0)


def _finalize_neutral(w: pd.DataFrame, data: MarketData, gross: float = 1.0) -> pd.DataFrame:
    """多空兜底：对齐、NaN→0、裁到 [-1, 1]，行绝对值和超 gross 者等比缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0).clip(lower=-1.0, upper=1.0))
    tot = out.abs().sum(axis=1)
    over = tot > float(gross) + 1e-15
    scale = pd.Series(1.0, index=out.index, dtype="float64")
    if bool(over.any()):
        scale[over] = float(gross) / tot[over]
    return out.mul(scale, axis=0)


def _beta_long_short_weights(beta: pd.DataFrame, data: MarketData, leg_frac: float,
                             leg_budget: float, beta_floor: float,
                             beta_cap: float) -> pd.DataFrame:
    """BAB 配权：低 beta 腿做多、高 beta 腿做空，腿内按 ``1/β`` 归一，美元中性。

    逐行（只用当期截面 beta）：
      1. 有效 beta 升序排名（并列按列索引稳定打破），多头腿取最低
         ``k = clip(ceil(leg_frac × n_valid), 1, n_valid // 2)`` 名、空头腿取最高
         ``k`` 名，中间分位**不下注**（经典 BAB 的三分位形态）；
      2. 腿内权重 ∝ ``1 / clip(β, beta_floor, beta_cap)``：beta 越低权重越大，
         使腿内每个资产的 beta 贡献 ``w_i·β_i`` 相等（等 beta 风险贡献）；低 beta 腿
         的组合 beta 天然低于高 beta 腿，美元中性下净 beta 为负（防御属性）；
      3. 两腿各归一到 ``leg_budget``（默认 0.5）→ 逐行和 ≈ 0（美元中性）、
         绝对值之和 = 2 × leg_budget ≤ 1（不加杠杆）；
      4. 有效 beta < 2 个（凑不出两腿）或预热期 → 整行空仓。
    """
    b = (beta.reindex(index=data.dates, columns=data.symbols)
         .apply(pd.to_numeric, errors="coerce"))
    vals = b.to_numpy(dtype="float64")
    valid = np.isfinite(vals)
    nv = valid.sum(axis=1).astype("float64")
    k = np.ceil(nv * float(leg_frac) - 1e-9)
    k = np.clip(k, 1.0, np.floor(np.maximum(nv, 0.0) / 2.0))
    k = np.where(nv >= 2.0, k, 0.0)
    rk = b.rank(axis=1, ascending=True, method="first",
                na_option="keep").to_numpy(dtype="float64")   # beta 最低 -> rank 1
    inv = 1.0 / np.clip(np.where(valid, vals, np.nan), float(beta_floor), float(beta_cap))
    inv = np.where(valid & np.isfinite(inv), inv, 0.0)
    on = k[:, None] > 0.0
    long_m = valid & on & (rk <= k[:, None])
    short_m = valid & on & (rk > (nv - k)[:, None])
    lraw = np.where(long_m, inv, 0.0)
    sraw = np.where(short_m, inv, 0.0)
    ls = lraw.sum(axis=1, keepdims=True)
    ss = sraw.sum(axis=1, keepdims=True)
    budget = float(leg_budget)
    w = (budget * np.where(ls > _EPS, lraw / np.where(ls > _EPS, ls, 1.0), 0.0)
         - budget * np.where(ss > _EPS, sraw / np.where(ss > _EPS, ss, 1.0), 0.0))
    return pd.DataFrame(np.clip(w, -1.0, 1.0), index=data.dates, columns=data.symbols)


def _inverse_beta_weights(beta: pd.DataFrame, data: MarketData, w_cap: float,
                          beta_floor: float, beta_cap: float) -> pd.DataFrame:
    """逆 beta 满仓配权：``w ∝ 1/clip(β, floor, cap)``，逐行归一到和 = 1。

    * beta 未知（预热期单个资产 NaN）→ 用**当期截面中位数**填补（视为中性资产）；
    * 整行 beta 都未知（预热期）→ 回退**等权 1/N**（满仓，不空仓）；
    * 单资产上限 ``max(w_cap, 1/N)`` 用 water-filling 再分配，避免近零 beta 资产
      吃满预算（防御策略本身也不该制造集中度风险）。
    """
    b = (beta.reindex(index=data.dates, columns=data.symbols)
         .apply(pd.to_numeric, errors="coerce"))
    vals = b.to_numpy(dtype="float64")
    n = max(int(vals.shape[1]), 1)
    clipped = np.clip(vals, float(beta_floor), float(beta_cap))   # NaN 保持 NaN
    raw = 1.0 / clipped
    finite = np.isfinite(raw)
    med = (pd.DataFrame(raw, index=b.index, columns=b.columns)
           .median(axis=1).to_numpy(dtype="float64"))             # 逐行截面中位数
    med = np.where(np.isfinite(med), med, 1.0)
    raw = np.where(finite, raw, med[:, None])
    any_valid = finite.any(axis=1)
    raw = np.where(any_valid[:, None], raw, 1.0)                  # 整行未知 -> 等权
    w = _normalize_rows(raw)
    cap = max(float(w_cap), 1.0 / float(n))
    w = _cap_water_fill(w, cap, iters=n + 4)
    return pd.DataFrame(w, index=data.dates, columns=data.symbols)


# --------------------------------------------------------------------------
# 策略 1：betting against beta（BAB，美元中性多空）
# --------------------------------------------------------------------------
class BettingAgainstBetaStrategy(Strategy):
    """BAB：滚动 beta 排序，做多低 beta 腿、做空高 beta 腿，腿内按 1/β 配权。"""

    name = "betting_against_beta"
    channel = "defensive"
    universe = "cross_section"
    long_only = False
    description = ("Betting-against-beta（自研版）：以等权组合为市场代理，逐资产滚动 OLS 估计 "
                   "beta 并朝当期截面均值收缩；每期按 beta 升序取最低 leg_frac 分位为多头腿、"
                   "最高 leg_frac 分位为空头腿（中间分位不下注），腿内权重 ∝ 1/clip(β) 后各自"
                   "归一到 0.5 预算——逐行权重和 ≈ 0（美元中性）、绝对值之和 = 1（不加杠杆）；"
                   "1/β 倾斜使腿内每个资产的 beta 贡献相等（w_i·β_i 恒定），而低 beta 腿的组合 beta "
                   "低于高 beta 腿，故美元中性下净 beta 为负——这正是 BAB 的防御属性。")
    hypothesis = ("成因（杠杆约束异象）：受杠杆/融资约束的投资者无法用「借钱买低 beta」放大收益，"
                  "只能退而买高 beta 资产来追求目标收益，于是高 beta 被系统性买贵、预期收益被压低，"
                  "低 beta 反而提供更高的单位风险收益；加上高 beta 资产多为『彩票型』题材股、散户"
                  "偏好与做空限制使其定价偏差更难被套利掉，做多做空两端即可赚到这条「beta 与预期"
                  "收益负相关」的截面价差。用 1/β 配权而非等权，是让腿内每个资产承担相等的 beta "
                  "贡献，避免单只极端 beta 资产主导整条腿的市场风险，价差因此更接近纯 beta 异象。"
                  "失效场景：强牛/流动性宽松阶段高 beta 暴力"
                  "反弹，空头腿亏损超过多头腿盈利（BAB 的最大回撤来源，且空头对被逼空标的风险不"
                  "封顶）；利率快速上行、风险偏好切换时防御风格整体跑输；beta 估计窗口与真实风险"
                  "结构错配（急转弯处滞后一个窗口）；等权市场是粗糙的单因子代理，真实因子多元时"
                  "残差仍被行业/风格共同因子污染；策略拥挤后价差被提前交易掉。")
    source = ("防御类范式——Betting Against Beta (Frazzini & Pedersen, 2014) 思路的本仓库原创"
              "实现：滚动 OLS beta + 截面收缩 + 三分位腿 + 1/β 腿内配权 + 美元中性归一，纯 "
              "numpy/pandas（不用 statsmodels）；与 regime 渠道 beta_timing（只做多的预算乘数"
              "再分配）、long_short 渠道 lowvol_ls（按波动率分位）在信号变量、配权规则与组合形态"
              "上均不同，所有工具函数在本模块内自实现。")
    params = {
        "beta_window": 60,        # 滚动 beta 估计窗口（日，约一个季度）
        "shrink": 0.2,            # beta 朝当期截面均值的收缩强度（0=不收缩）
        "leg_frac": 1.0 / 3.0,    # 多空两腿各取的分位（中间分位不下注）
        "leg_budget": 0.5,        # 单腿预算：多头 +0.5、空头 -0.5 -> 行和 ≈ 0
        "beta_floor": 0.10,       # beta 下限：≤0.10（含负 beta）视为最强防御，1/β 不爆表
        "beta_cap": 4.0,          # beta 上限：避免极高 beta 资产权重被压成 0 而失去区分度
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        beta = _estimated_beta(data, int(p["beta_window"]), float(p["shrink"]))
        w = _beta_long_short_weights(beta, data, float(p["leg_frac"]),
                                     float(p["leg_budget"]), float(p["beta_floor"]),
                                     float(p["beta_cap"]))
        return _finalize_neutral(w, data, gross=2.0 * float(p["leg_budget"]))


# --------------------------------------------------------------------------
# 策略 2：低 beta 择时（只做多、满仓的逆 beta 配置）
# --------------------------------------------------------------------------
class LowBetaTimingStrategy(Strategy):
    """低 beta 倾斜配置：权重 ∝ 1/β_shrunk，逐行归一到和 = 1（满仓、只做多）。"""

    name = "low_beta_timing"
    channel = "defensive"
    universe = "cross_section"
    long_only = True
    description = ("低 beta 倾斜配置（只做多满仓）：逐资产滚动 OLS 估计对等权市场的 beta 并朝当期"
                   "截面均值收缩，权重 ∝ 1/clip(β, floor, cap)——beta 越低权重越高，逐行归一到"
                   "和 = 1（满仓、无杠杆），并对单资产权重设 max(w_cap, 1/N) 上限用 water-filling "
                   "再分配；单个资产 beta 未知时用当期截面中位数填补，整行未知（预热期）回退等权 "
                   "1/N。组合 beta 因此低于等权基准，是纯『低 beta 倾斜』的风险端配置。")
    hypothesis = ("成因：与 BAB 同源的杠杆约束异象，但只用多头表达——在不能做空或不愿承担空头"
                  "尾部风险时，把预算按 1/β 重新分配即可把组合 beta 压到等权基准之下，用更少的"
                  "系统性风险换取相近（甚至更高）的收益，夏普与最大回撤同时改善；beta 的可预测性"
                  "远强于收益，故这是一种『不预测收益、只管理风险』的稳健倾斜。朝截面均值收缩 beta "
                  "可抑制估计噪声导致的排序抖动，单资产上限避免近零 beta 资产垄断预算。失效场景："
                  "高 beta 领涨的强牛阶段（成长/题材行情）持续跑输基准；各资产 beta 都贴近 1 的"
                  "同质化行情中倾斜失去区分度（退化为等权）；beta 结构突变处窗口估计滞后；低 beta "
                  "资产拥挤、估值被买贵后异象衰减；等权市场代理粗糙时『低 beta』可能只是低行业暴露。")
    source = ("防御类范式——低 beta 倾斜配置 (low-beta / minimum-beta tilt) 的本仓库原创实现；"
              "与 regime 渠道 beta_timing（等权预算 × 乘数 g=clip(1+k(1-β))，靠 beta 均值恒等式"
              "保持行和）不同，此处是**逆 beta 直接配权 + 截面收缩 + 上限 water-filling**；与 "
              "allocation 渠道 inverse_vol / risk_parity_alloc（按波动/协方差分配）的输入变量也不同。")
    params = {
        "beta_window": 60,        # 滚动 beta 估计窗口（日）
        "shrink": 0.2,            # beta 朝当期截面均值的收缩强度
        "beta_floor": 0.20,       # beta 下限（≤0.20 视为最强防御，1/β 最大 5）
        "beta_cap": 2.50,         # beta 上限（≥2.50 视为最高风险，1/β 最小 0.4）
        "w_cap": 0.40,            # 单资产权重上限（超出部分 water-filling 再分配）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        beta = _estimated_beta(data, int(p["beta_window"]), float(p["shrink"]))
        w = _inverse_beta_weights(beta, data, float(p["w_cap"]),
                                  float(p["beta_floor"]), float(p["beta_cap"]))
        return _finalize_long(w, data)


# --------------------------------------------------------------------------
# 策略 3：防御质量复合（低波动 + 低回撤 + 慢均线之上）
# --------------------------------------------------------------------------
class DefensiveQualityStrategy(Strategy):
    """防御质量复合打分：低波动 + 浅回撤 + 正长期趋势，做多得分最高的一篮子。"""

    name = "defensive_quality"
    channel = "defensive"
    universe = "cross_section"
    long_only = True
    description = ("防御质量复合（只做多满仓）：三个纯价格成分先各自截面标准化再合成——(1) 低波动："
                   "−已实现波动（vol_window）；(2) 浅回撤：−当期回撤深度（1 − p/滚动 dd_window 期"
                   "高点）；(3) 正长期趋势：p/慢均线(ma_window) − 1（在慢均线之上为正，裁剪到 "
                   "±trend_clip 抑制离群）；合成分 = w_vol·z₁ + w_dd·z₂ + w_trend·z₃，按**排名线性**"
                   "配权做多得分最高的 top_frac 一篮子（第 j 名权重 ∝ k+1−j），逐行归一到和 = 1；"
                   "任一成分未知的资产当日不选，预热期整行空仓。")
    hypothesis = ("成因：三个成分刻画同一类『防御质量』资产的不同侧面——低波动对应彩票偏好与杠杆"
                  "约束导致的高波资产被高估；浅回撤说明价格仍被资金承接、下行弹性小（贴近自身高点"
                  "的资产抛压已消化）；站上慢均线说明长期趋势为正、不是在下跌途中『看似安全』。"
                  "三者合成能过滤单因子的假信号：低波但深回撤者（阴跌的类债券资产）会被回撤与趋势"
                  "成分否掉，趋势向上但剧烈震荡者会被波动成分否掉，剩下的是『又稳又贴近高点又在长期"
                  "上升通道』的一篮子，长期回撤更小、夏普更高。用截面 z 合成保证量纲统一、互不淹没，"
                  "用排名线性配权保证对离群值稳健。失效场景：三成分高度相关（危机中一起恶化），合成"
                  "退化为单因子、区分度骤降；风格急切换/深 V 反弹时深回撤资产弹性最大，防御篮子踏空；"
                  "趋势成分在拐点处滞后（均线之上但已见顶）；成分权重是主观先验，相关性结构变化会让"
                  "某一维主导；防御篮子拥挤后估值溢价被提前交易掉。")
    source = ("防御类范式——『低波动 + 浅回撤 + 长期趋势』复合质量打分的本仓库原创实现（纯价格构造，"
              "不臆造基本面/盈利质量数据）；与 long_short 渠道 quality_ls（价格效率 + 低波动、两端"
              "下注）、factor 渠道 low_volatility（波动单因子）在成分、配权规则（排名线性）与组合"
              "形态（只做多满仓）上均不同。")
    params = {
        "vol_window": 21,         # 已实现波动窗口（约一个月）
        "dd_window": 63,          # 回撤/滚动高点窗口（约一个季度）
        "ma_window": 126,         # 长期趋势慢均线窗口（约半年）
        "trend_clip": 0.30,       # 趋势成分裁剪（±30%），抑制离群值主导合成分
        "w_vol": 0.40,            # 低波动成分权重
        "w_dd": 0.35,             # 浅回撤成分权重
        "w_trend": 0.25,          # 正趋势成分权重
        "top_frac": 1.0 / 3.0,    # 做多篮子的分位（1/3）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        px = _panel(data)
        r = _returns(px)
        vol = _realized_vol(r, int(p["vol_window"]), data.periods_per_year)
        dd = _current_drawdown(px, int(p["dd_window"]))
        ma = ind.sma(px, max(int(p["ma_window"]), 2))
        trend = (px / ma.where(ma > _EPS) - 1.0).clip(-float(p["trend_clip"]),
                                                     float(p["trend_clip"]))
        score = (float(p["w_vol"]) * _row_z(-vol)
                 - float(p["w_dd"]) * _row_z(dd)
                 + float(p["w_trend"]) * _row_z(trend))
        w = _rank_top_weights(score, data, float(p["top_frac"]))
        return _finalize_long(w, data)


# --------------------------------------------------------------------------
# 策略 4：回撤规避（逐资产回撤闸门 + 低波倾斜 + 组合层健康度缩放）
# --------------------------------------------------------------------------
class DrawdownAverseStrategy(Strategy):
    """回撤规避：深回撤资产权重被压低/剔除，偏好贴近滚动高点且低波的资产。"""

    name = "drawdown_averse"
    channel = "defensive"
    universe = "timing"
    long_only = True
    description = ("回撤规避（只做多，行和 ≤ 1）：逐资产计算当期回撤深度 dd = 1 − p/滚动 dd_window "
                   "期高点，生存度 survive = clip(1 − dd/dd_tol, 0, 1)——贴近自身滚动高点者 ≈ 1、"
                   "回撤达 dd_tol 者被**完全剔除**；再乘低波倾斜 1/clip(已实现波动)，在存活资产间"
                   "归一；组合层按截面中位回撤做健康度缩放 expo = clip(1 − median(dd)/dd_tol, "
                   "expo_floor, 1)，即『越多的资产处于深回撤 → 整体越退向现金』。最终权重 = expo × "
                   "截面归一权重，逐行和 ≤ 1（差额为现金）；预热期（回撤/波动未知）整行空仓。")
    hypothesis = ("成因：回撤有两重防御价值——(1) 数学上，亏损的复利不对称（跌 40% 需涨 67% 才回"
                  "本），压制深回撤资产的权重直接改善几何收益与最大回撤；(2) 行为/信息上，持续远离"
                  "自身高点意味着抛压尚未消化、负面信息仍在扩散（动量的下行延续），而贴近滚动高点的"
                  "资产说明买盘承接充分、趋势健康；叠加低波倾斜进一步回避彩票型标的。组合层的健康度"
                  "缩放把『回撤的普遍性』当成系统性风险温度计：普遍深回撤时整体退向现金，而不是在"
                  "矮子里拔将军继续满仓。失效场景：V 型反转——深回撤资产反弹弹性最大，剔除它们会"
                  "系统性踏空（回撤信号的滞后性在拐点最贵）；震荡市中信号在 dd_tol 附近反复进出，"
                  "换手与成本上升；长期慢牛中 dd 普遍很小，闸门几乎不生效（退化为纯逆波动配置）；"
                  "系统性危机中所有资产同时深回撤 → 全现金，虽保住本金却完全放弃后续修复收益。")
    source = ("防御类范式——回撤规避/回撤控制 (drawdown-averse allocation) 的本仓库原创实现；与 "
              "riskmgmt 渠道 drawdown_throttle / cppi / circuit_breaker（都是对**等权基准净值**做"
              "组合层降杠杆的 overlay）不同，此处是**逐资产**回撤闸门 + 截面低波配权 + 组合层健康度"
              "缩放三层结构；与 momentum 渠道 high_52w（把『接近新高』当动量突破做多信号）在信号"
              "语义（规避 vs 追涨）、配权（连续门限 vs 等预算）与现金处理上均不同。")
    params = {
        "dd_window": 63,          # 滚动高点/回撤窗口（约一个季度）
        "dd_tol": 0.20,           # 回撤容忍度：dd ≥ 20% -> 权重清零
        "vol_window": 21,         # 低波倾斜的已实现波动窗口（约一个月）
        "vol_floor": 0.02,        # 年化波动下限（避免近零波动资产垄断预算）
        "vol_cap": 3.00,          # 年化波动上限（倾斜不至于被极高波资产拉平）
        "expo_floor": 0.0,        # 组合层健康度缩放的下限（0 = 允许全现金）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        px = _panel(data)
        r = _returns(px)
        dd_tol = max(float(p["dd_tol"]), 1e-6)
        dd = _current_drawdown(px, int(p["dd_window"]))
        survive = (1.0 - dd / dd_tol).clip(lower=0.0, upper=1.0)     # NaN 原样保留
        vol = _realized_vol(r, int(p["vol_window"]), data.periods_per_year)
        tilt = 1.0 / vol.clip(lower=float(p["vol_floor"]), upper=float(p["vol_cap"]))
        raw = (survive * tilt).to_numpy(dtype="float64")             # 预热期 NaN -> 0
        expo = (1.0 - dd.median(axis=1) / dd_tol).clip(
            lower=float(p["expo_floor"]), upper=1.0).fillna(0.0).to_numpy(dtype="float64")
        w = expo[:, None] * _normalize_rows(raw)
        return _finalize_long(pd.DataFrame(w, index=data.dates, columns=data.symbols), data)


# 便于外部（报告/研究记录）按渠道枚举的辅助常量
CHANNEL = "defensive"
STRATEGY_NAMES = (
    "betting_against_beta",
    "low_beta_timing",
    "defensive_quality",
    "drawdown_averse",
)

__all__ = [
    "BettingAgainstBetaStrategy",
    "LowBetaTimingStrategy",
    "DefensiveQualityStrategy",
    "DrawdownAverseStrategy",
    "CHANNEL",
    "STRATEGY_NAMES",
]
