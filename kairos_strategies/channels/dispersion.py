"""渠道：dispersion —— 截面分散度与相关性状态策略。

本渠道聚焦「截面二阶矩结构」这一独特维度：不看单一资产自身的波动水平
（volatility 渠道），不看个别资产对/残差的错价（statarb / pairs 渠道），
也不做 bull/bear 方向状态识别（regime 渠道），而是度量**资产之间**的
分化与同步程度：

  - ``dispersion_timing``   —— 分散度择时：滚动**截面收益分散度**（各资产日收益的
    截面标准差，用平均已实现波动归一成无量纲比率 DR）度量市场分化程度；高分散
    （个股走势分化大，选股空间大）时把预算倾斜给「截面均值回归/选股腿」（买近期
    相对弱者），低分散（齐涨齐跌，指数化更优）时倾斜给「时序动量/指数腿」。这是
    dispersion trading 思想在纯多头日频上的概念性适配（详见类 docstring）。
  - ``correlation_regime``  —— 相关性状态去杠杆：滚动估计**平均两两相关系数**
    （精确的成对 Pearson 相关，滚动窗口内 E[xy]−E[x]E[y] 形式向量化计算），
    相关性飙升（危机同步、分散化失效）时按分段线性映射降低总敞口（risk-off），
    相关性低时满仓等权分散（等权 × 敞口缩放）。
  - ``vol_dispersion_ls``   —— 波动率分散度多空代理：做空「等权指数篮子」、做多
    「逆波动单腿分散篮子」，用「平均单股波动 / 指数波动」比率 D 门控仓位规模，
    D 高（单股波动远高于指数波动 = 高分散）时满强度、D ≈ 1（完全同步）时平仓；
    美元中性（行和 ≈ 0、绝对值和 ≤ 1）。这是对 index-vol vs single-stock-vol
    分散度交易的日频线性代理（详见类 docstring）。

三个策略均为本仓库 100% 原创实现，仅用 numpy/pandas，离线且确定性（无随机数）。

防未来函数：第 t 期权重只使用截至 t（含 t 收盘）的收益——截面分散度、平均两两
相关、已实现波动全部是 trailing 滚动窗口统计（``min_periods=window``，样本不足
即 NaN）；预热期按规范回退：``dispersion_timing`` / ``correlation_regime`` 回退
等权（分散度/相关性未知时持有最分散的等权组合），``vol_dispersion_ls`` 回退空仓
（分散度未确认不下注）。引擎还会再滞后一期，双重保险。

权重规则：long_only 策略（前两个）权重 ∈ [0,1] 且每行和 ≤ 1；多空策略
（``vol_dispersion_ls``）每行和 ≈ 0（美元中性）且绝对值和 ≤ 1（不加杠杆）。
``_finalize_long`` / ``_finalize_neutral`` 做最后一道对齐、裁剪与缩放兜底。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-12
_TINY = 1e-12


# --------------------------------------------------------------------------
# 模块内小工具（全部自实现，不从其它渠道 import 私有函数）
# --------------------------------------------------------------------------
def _returns(data: MarketData) -> pd.DataFrame:
    """日收益面板（首行保持 NaN，让滚动窗口的 min_periods 自然处理预热期）。"""
    return data.prices.astype("float64").pct_change()


def _xs_dispersion(r: pd.DataFrame, periods_per_year: int) -> pd.Series:
    """逐日**截面收益分散度**：当日各资产收益的截面标准差（ddof=1），年化。

    只用到当期截面（≤ t），度量「今天资产之间分化有多大」。
    """
    return r.std(axis=1, ddof=1) * np.sqrt(float(periods_per_year))


def _realized_vol_panel(r: pd.DataFrame, window: int,
                        periods_per_year: int) -> pd.DataFrame:
    """逐资产 trailing 已实现波动（年化），窗口不足 → NaN。"""
    w = max(int(window), 2)
    return r.rolling(w, min_periods=w).std() * np.sqrt(float(periods_per_year))


def _dispersion_state(r: pd.DataFrame, periods_per_year: int, disp_window: int,
                      vol_window: int, vol_floor: float, ratio_lo: float,
                      ratio_hi: float) -> pd.Series:
    """把截面分散度映射为无量纲状态 d ∈ [0,1]（NaN = 预热期未知）。

    DR_t = 平滑后的截面分散度（disp_window 日均值，年化） / 平均单资产已实现波动。
    DR 无量纲：齐涨齐跌（收益完全同步）时截面标准差 ≈ 0 → DR ≈ 0；资产各自独立
    波动时截面标准差 ≈ 平均波动 → DR ≈ 1；走势持续分化（如强者恒强、弱者恒弱）时
    DR 可远大于 1。d = clip((DR − ratio_lo) / (ratio_hi − ratio_lo), 0, 1)：
    d → 0 低分散（指数/趋势环境），d → 1 高分散（选股/均值回归环境）。
    """
    dw = max(int(disp_window), 1)
    disp = _xs_dispersion(r, periods_per_year)
    disp_s = disp.rolling(dw, min_periods=dw).mean() if dw > 1 else disp
    avg_vol = _realized_vol_panel(r, vol_window, periods_per_year).mean(axis=1)
    dr = disp_s / avg_vol.clip(lower=float(vol_floor))
    lo, hi = float(ratio_lo), float(max(ratio_hi, ratio_lo + _TINY))
    return ((dr - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def _avg_pairwise_corr(r: pd.DataFrame, window: int) -> pd.Series:
    """滚动**平均两两相关系数** ρ̂_t ∈ [-1,1]（NaN = 窗口不足或全部退化）。

    对每一对 (i, j)，在 trailing window 上用「二阶矩差」形式向量化计算 Pearson
    相关：cov = E[xy] − E[x]E[y]，var = E[x²] − E[x]²（同一窗口同一 ddof 约定，
    比值与 ddof 无关），再对全部**有效对**取平均（某资产在窗口内零方差时其相关
    对不可估，跳过而不污染其它对）。第 t 期只用 [t−window+1, t] 的收益，严格因果。
    """
    cols = list(r.columns)
    n = len(cols)
    if n < 2:
        return pd.Series(np.nan, index=r.index, dtype="float64")
    w = max(int(window), 3)
    mu = r.rolling(w, min_periods=w).mean()
    exx = r.pow(2).rolling(w, min_periods=w).mean()
    var = (exx - mu.pow(2)).clip(lower=0.0)
    sd = np.sqrt(var)
    acc = pd.Series(0.0, index=r.index, dtype="float64")
    cnt = pd.Series(0.0, index=r.index, dtype="float64")
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cols[i], cols[j]
            exy = (r[ci] * r[cj]).rolling(w, min_periods=w).mean()
            cov = exy - mu[ci] * mu[cj]
            den = sd[ci] * sd[cj]
            c = (cov / den.where(den > _TINY)).clip(-1.0, 1.0)
            acc = acc + c.fillna(0.0)
            cnt = cnt + c.notna().astype("float64")
    out = acc / cnt.where(cnt > 0.0)
    return out.clip(-1.0, 1.0)


def _inverse_vol_weights(r: pd.DataFrame, window: int, periods_per_year: int,
                         vol_floor: float) -> pd.DataFrame:
    """trailing 逆波动加权（行和 = 1）：1/σ_i 归一；波动不可估的行回退等权。"""
    n = max(int(r.shape[1]), 1)
    vol = _realized_vol_panel(r, window, periods_per_year).clip(lower=float(vol_floor))
    inv = 1.0 / vol
    tot = inv.sum(axis=1)
    ivw = inv.div(tot.where(tot > _EPS), axis=0)
    return ivw.where(ivw.notna(), 1.0 / float(n))


def _vol_dispersion_size(r: pd.DataFrame, periods_per_year: int, vol_window: int,
                         size_lo: float, size_hi: float,
                         vol_floor: float) -> pd.Series:
    """波动率分散度门控 s ∈ [0,1]：D = 平均单股波动 / 等权指数波动。

    指数（等权每日再平衡）波动因分散化而低于平均单股波动：完全同步时 D ≈ 1
    （分散化收益消失），资产独立时 D ≈ √N × rms/mean ≥ 1（高分散）。
    s = clip((D − size_lo) / (size_hi − size_lo), 0, 1)，预热期（D 未知）→ 0（空仓）。
    """
    w = max(int(vol_window), 2)
    r_idx = r.mean(axis=1)
    vol_idx = (r_idx.rolling(w, min_periods=w).std()
               * np.sqrt(float(periods_per_year))).clip(lower=float(vol_floor))
    vol_ss = _realized_vol_panel(r, w, periods_per_year).mean(axis=1)
    d = vol_ss / vol_idx
    lo, hi = float(size_lo), float(max(size_hi, size_lo + _TINY))
    return ((d - lo) / (hi - lo)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _equal_weight_panel(data: MarketData) -> pd.DataFrame:
    """等权面板（每资产 1/N），用作预热期回退。"""
    n = max(int(data.n_assets), 1)
    return pd.DataFrame(1.0 / float(n), index=data.dates, columns=data.symbols)


def _broadcast(scale: pd.Series, data: MarketData) -> pd.DataFrame:
    """组合层敞口序列 → 等权权重面板（每资产 scale/N，行和 = scale）。"""
    n = max(int(data.n_assets), 1)
    per = scale.reindex(data.dates).fillna(0.0).to_numpy(dtype="float64") / float(n)
    return pd.DataFrame(np.repeat(per[:, None], n, axis=1),
                        index=data.dates, columns=data.symbols)


def _finalize_long(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、NaN → 0、裁到 [0, cap]，行和超 cap 的行等比缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0).clip(lower=0.0, upper=cap))
    total = out.sum(axis=1)
    factor = (cap / total.where(total > _EPS)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


def _finalize_neutral(w: pd.DataFrame, data: MarketData,
                      budget: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、NaN → 0、裁到 [-1, 1]，行绝对值和超 budget 的行等比缩回。

    注意：缩放行会同时等比缩小行和（保持 ≈ 0 的美元中性不被破坏）。
    """
    out = (w.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0).clip(lower=-1.0, upper=1.0))
    gross = out.abs().sum(axis=1)
    factor = (budget / gross.where(gross > _EPS)).where(gross > budget, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


# --------------------------------------------------------------------------
# 策略 1：分散度择时（dispersion trading 思想的纯多头日频适配）
# --------------------------------------------------------------------------
class DispersionTimingStrategy(Strategy):
    """分散度择时：高分散倾斜「选股/均值回归腿」，低分散倾斜「指数/趋势腿」。"""

    name = "dispersion_timing"
    channel = "dispersion"
    universe = "cross_section"
    long_only = True
    description = ("分散度择时：用滚动截面收益分散度（各资产日收益的截面标准差，经平均已实现"
                   "波动归一为无量纲比率 DR，再线性映射为状态 d ∈ [0,1]）度量市场分化程度——"
                   "高分散（d → 1，个股走势分化大）时预算倾斜给选股/均值回归腿（按近 rev_window "
                   "日收益的截面 z 反向配权，做多相对弱者），低分散（d → 0，齐涨齐跌）时倾斜给"
                   "指数/趋势腿（时序动量为正的资产等预算 1/N 持有）；两腿按 d 连续混合，"
                   "预热期（分散度未知）回退等权，行和 ≤ 1、权重 ≥ 0。")
    hypothesis = ("核心假设：截面分散度是『选股能力回报』的度量——分散度越高，资产走势越分化，"
                  "截面相对定价的信息含量与均值回归强度越大（过度反应更容易在分化环境中被纠正），"
                  "此时做多相对弱者的截面回归腿占优；分散度极低时资产被同一系统性因子驱动"
                  "（齐涨齐跌），截面选择没有增益，跟随指数/时序趋势更有效。为什么分散度可度量"
                  "风险与选股能力：它正是『特质波动占比』的价格侧影像——截面标准差大而个体波动小"
                  "意味着收益来自可分辨的个体信息而非共同噪声。失效场景：高分散由持续性结构分化"
                  "（强者恒强的行业行情）驱动时，均值回归腿会持续接刀跑输截面动量；低分散的"
                  "单边牛市中趋势腿的 0/1 动量开关会频繁进出；分散度状态用自身波动水平归一，"
                  "波动长期漂移时 DR 的锚会缓慢移动。")
    source = ("指数/单股分散度交易 (dispersion trading) 与适应性市场假说的『状态 × 子策略』"
              "范式的纯多头日频概念性适配；截面分散度归一（DR 比率）、两腿构造与连续混合"
              "均为本仓库原创实现，与 regime 渠道的状态机切换（ER 判别趋势/震荡）在状态变量"
              "与子逻辑上均不同。")
    params = {
        "disp_window": 10,    # 截面分散度的平滑窗口（日）
        "vol_window": 40,     # 归一用的逐资产已实现波动窗口（日）
        "vol_floor": 0.01,    # 年化波动下限，避免近零波动把 DR 除爆
        "ratio_lo": 0.30,     # DR ≤ ratio_lo → 低分散（d = 0，纯指数/趋势腿）
        "ratio_hi": 1.00,     # DR ≥ ratio_hi → 高分散（d = 1，纯选股/均值回归腿）
        "mom_window": 60,     # 趋势腿：时序动量回看窗口（日）
        "rev_window": 5,      # 回归腿：截面相对强弱回看窗口（日）
        "z_cap": 2.0,         # 回归腿截面 z 截断，防单一离群资产吃满预算
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        n = int(data.n_assets)
        if n < 2:
            return pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        close = data.prices.astype("float64")
        r = _returns(data)
        # 分散度状态 d ∈ [0,1]（trailing 统计，NaN = 预热期）
        d = _dispersion_state(r, data.periods_per_year, int(p["disp_window"]),
                              int(p["vol_window"]), float(p["vol_floor"]),
                              float(p["ratio_lo"]), float(p["ratio_hi"]))
        # 指数/趋势腿：时序动量为正 → 等预算 1/N 持有（低分散时占优）
        mom = close.pct_change(int(p["mom_window"]))
        trend_sig = ((mom > 0.0) & mom.notna()).astype("float64")
        w_trend = trend_sig / float(n)
        # 选股/均值回归腿：近 rev_window 日收益的截面 z，做多相对弱者（高分散时占优）
        rr = close.pct_change(int(p["rev_window"]))
        mu = rr.mean(axis=1)
        sd = rr.std(axis=1)
        z = rr.sub(mu, axis=0).div(sd.where(sd > _TINY), axis=0)
        z = z.clip(-float(p["z_cap"]), float(p["z_cap"]))
        raw = (-z).clip(lower=0.0)                       # 只给相对弱者正预算
        tot = raw.sum(axis=1)
        w_rev = raw.div(tot.where(tot > _TINY), axis=0)
        # 回归腿退化（截面无分化/预热）→ 该腿回退等权
        ok = np.broadcast_to(tot.gt(_TINY).to_numpy()[:, None], w_rev.shape)
        w_rev = w_rev.where(ok, 1.0 / float(n))
        w_rev = w_rev.fillna(1.0 / float(n))
        # 连续混合；分散度状态未知（预热）→ 整体回退等权
        dv = d.to_numpy(dtype="float64")
        blend = ((1.0 - dv)[:, None] * w_trend.to_numpy(dtype="float64")
                 + dv[:, None] * w_rev.to_numpy(dtype="float64"))
        out = np.where(np.isfinite(dv)[:, None], blend, 1.0 / float(n))
        return _finalize_long(pd.DataFrame(out, index=data.dates,
                                           columns=data.symbols), data)


# --------------------------------------------------------------------------
# 策略 2：相关性状态去杠杆（危机同步 → risk-off）
# --------------------------------------------------------------------------
class CorrelationRegimeStrategy(Strategy):
    """相关性状态去杠杆：平均两两相关飙升时降低总敞口，相关性低时满仓等权分散。"""

    name = "correlation_regime"
    channel = "dispersion"
    universe = "timing"
    long_only = True
    description = ("相关性状态去杠杆：滚动 corr_window 期精确估计全部资产对的平均两两相关系数 "
                   "ρ̂（E[xy]−E[x]E[y] 形式向量化，逐对 Pearson 相关取均值），把 ρ̂ 经分段线性"
                   "映射为总敞口 scale ∈ [expo_min, 1]：ρ̂ ≤ rho_lo（低相关，分散化有效）满仓"
                   "等权，ρ̂ ≥ rho_hi（相关性飙升，危机同步）降到 expo_min，中间线性过渡；"
                   "权重 = 等权(1/N) × scale（行和 = scale ≤ 1），预热期（ρ̂ 未知）回退等权满仓。")
    hypothesis = ("核心假设：平均两两相关性是系统性风险同步程度的直接度量——正常市场中资产"
                  "由各自的特质信息驱动，相关性低，等权分散能以低组合波动持有全部风险溢价；"
                  "危机/去杠杆事件中『所有资产相关性趋近 1』，分散化收益消失，同等名义仓位承担"
                  "的组合波动与尾部回撤成倍放大，而单位风险补偿并未同步升高，因此按相关性状态"
                  "缩放敞口可以改善风险调整后收益并显著压低尾部回撤。相关性（而非波动率）作为"
                  "状态变量的优点：它刻画的是组合层面的『分散化还剩多少』，与组合波动互补而不"
                  "重复。失效场景：高相关但持续上涨的『良性同步牛市』中被减仓踏空；相关性在"
                  "崩盘当天才跳升，滚动窗口使减仓滞后一个窗口；rho_lo/rho_hi 绝对锚点在不同"
                  "资产类别（天然高相关的同行业股票 vs 跨资产）间需要重标定。")
    source = ("相关性 regime / crisis-contagion 去杠杆范式（『correlations go to 1 in a "
              "crisis』）的纯价格日频实现；滚动平均两两相关的向量化估计、分段线性敞口映射"
              "与等权 × 缩放的组合形态均为本仓库原创，与 regime 渠道（波动分位/EM/beta 状态）"
              "的状态变量完全不同。")
    params = {
        "corr_window": 60,    # 平均两两相关的滚动估计窗口（日）
        "rho_lo": 0.30,       # ρ̂ ≤ rho_lo → 满仓（scale = 1）
        "rho_hi": 0.85,       # ρ̂ ≥ rho_hi → 深度去杠杆（scale = expo_min）
        "expo_min": 0.15,     # 极端高相关时的敞口下限（保留少量分散化持仓）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        if int(data.n_assets) < 2:
            return pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        rho = _avg_pairwise_corr(_returns(data), int(p["corr_window"]))
        lo, hi = float(p["rho_lo"]), float(p["rho_hi"])
        u = ((hi - rho) / max(hi - lo, _TINY)).clip(lower=0.0, upper=1.0)
        expo_min = float(p["expo_min"])
        scale = (expo_min + (1.0 - expo_min) * u).clip(lower=0.0, upper=1.0)
        scale = scale.fillna(1.0)                # 预热期（ρ̂ 未知）→ 等权满仓回退
        return _finalize_long(_broadcast(scale, data), data)


# --------------------------------------------------------------------------
# 策略 3：波动率分散度多空代理（long 分散篮子 / short 指数篮子）
# --------------------------------------------------------------------------
class VolDispersionLsStrategy(Strategy):
    """波动率分散度多空代理：做空等权指数篮子、做多逆波动单腿分散篮子，D 门控规模。"""

    name = "vol_dispersion_ls"
    channel = "dispersion"
    universe = "cross_section"
    long_only = False
    description = ("波动率分散度多空代理：每条腿各占 leg_budget=0.5 预算——做空『等权指数篮子』"
                   "（每资产 −0.5/N，其波动含分散化收益、系统性同步成分被短路），做多『逆波动"
                   "加权的单腿分散篮子』（+0.5×ivw_i，低波动资产权重更高，代表可分散的个体"
                   "风险敞口）；仓位规模 s ∈ [0,1] 由波动率分散度比率 D = 平均单股已实现波动 / "
                   "等权指数已实现波动 门控：D ≥ size_hi（单股波动远高于指数波动 = 高分散，"
                   "分散化收益丰厚）满强度，D ≤ size_lo（同步，分散度消失）平仓；净权重 = "
                   "s×0.5×(ivw_i − 1/N)，行和 ≈ 0（美元中性）、绝对值和 ≤ 1，预热期空仓。")
    hypothesis = ("核心假设（dispersion trading 的日频线性代理）：经典的分散度交易做多单股"
                  "波动、做空指数波动，其收益来源是『单股已实现方差 − 指数已实现方差』即波动率"
                  "分散度溢价——指数波动因相关性分散化而系统性低于平均单股波动，卖出指数波动/"
                  "买入单股波动等于收割这一结构性差价。纯价格线性持仓无法复制期权的方差凸性，"
                  "本策略用『逆波动多头篮子 vs 等权指数空头』的市场中性 book 作代理：多腿按 1/σ "
                  "加权把预算集中在个体风险上、空腿均匀短路同步成分，且仅当 D = σ_单股均值/"
                  "σ_指数 确认高分散状态（单股波动远高于指数波动）时才开仓——此时截面个体波动"
                  "主导、篮子间价差的信息含量最高；相关性趋近 1 时 D → 1，分散化收益消失，"
                  "自动平仓。失效场景：线性代理赚的实际是『低波动腿相对高波动腿』的截面差价，"
                  "当高波动资产领涨（投机性逼空、垃圾股暴动）时持续亏损；D 的绝对锚点在天然"
                  "低相关的 universe（D 恒高）上退化为常开的低波动多空 book；波动率估计窗口"
                  "滞后于分散度状态的突然塌缩。")
    source = ("指数波动 vs 单股波动分散度交易 (index-vol / single-stock-vol dispersion "
              "trade) 的日频纯价格线性代理；『逆波动分散篮子 vs 等权指数篮子』的双腿构造与 "
              "D 比率门控为本仓库原创实现，与 statarb 渠道（错价回归型多空）、long_short 渠道"
              "（因子排序型多空）的信号来源均不同。")
    params = {
        "vol_window": 60,     # 已实现波动的滚动估计窗口（日）
        "vol_floor": 0.01,    # 年化波动下限，避免近零波动污染 D 与 ivw
        "size_lo": 1.10,      # D ≤ size_lo → 平仓（分散化收益太薄）
        "size_hi": 1.60,      # D ≥ size_hi → 满强度（高分散状态确认）
        "leg_budget": 0.5,    # 单腿预算：多头腿 +0.5、空头腿 −0.5 → 绝对值和 ≤ 1
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        n = int(data.n_assets)
        if n < 2:
            return pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        r = _returns(data)
        # 波动率分散度门控 s（预热期 D 未知 → 0 → 空仓）
        s = _vol_dispersion_size(r, data.periods_per_year, int(p["vol_window"]),
                                 float(p["size_lo"]), float(p["size_hi"]),
                                 float(p["vol_floor"]))
        # 多头分散腿：逆波动加权（行和 1）；波动不可估的行已在辅助函数内回退等权
        ivw = _inverse_vol_weights(r, int(p["vol_window"]), data.periods_per_year,
                                   float(p["vol_floor"]))
        leg = float(p["leg_budget"])
        sv = s.to_numpy(dtype="float64")[:, None]
        # 净权重 = s × [leg × ivw（多分散腿） − leg × 1/N（空等权指数腿）]
        w = sv * leg * (ivw.to_numpy(dtype="float64") - 1.0 / float(n))
        return _finalize_neutral(pd.DataFrame(w, index=data.dates,
                                              columns=data.symbols), data)


# 便于外部（报告/研究记录）按渠道枚举的辅助常量
CHANNEL = "dispersion"
STRATEGY_NAMES = (
    "dispersion_timing",
    "correlation_regime",
    "vol_dispersion_ls",
)

__all__ = [
    "DispersionTimingStrategy",
    "CorrelationRegimeStrategy",
    "VolDispersionLsStrategy",
    "CHANNEL",
    "STRATEGY_NAMES",
]
