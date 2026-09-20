"""渠道：regime —— 市场状态识别与切换策略（显式状态识别 + 按状态切换子逻辑）。

与既有渠道的分工（避免重复）：
  - ``volatility.vol_regime_filter`` 是**二元**波动状态开关（双分位状态机，满仓/空仓）；
    本渠道 ``regime_vol_timing`` 用**多档分位锚点 + 连续缩放**（低波满仓、中高波按
    分位线性降杠杆），是「状态强度」而非「状态开关」。
  - ``taa.trend_regime_taa`` 是组合层「指数 vs 长期均线」的 risk-on/off 切换；
    本渠道 ``trend_regime_switch`` 是**逐资产**「趋势 vs 震荡」状态机，并在两套
    **子逻辑**（动量持有 / 超跌买入）之间切换——状态决定的不是仓位有无，而是策略本身。
  - ``em_regime``：自研 2 分量一维高斯 EM（numpy 实现、确定性初始化、无随机数），
    walk-forward 只用截至当期的历史收益拟合，bull/bear 后验概率连续决定总敞口。
  - ``beta_timing``：滚动 beta 状态择时，逐资产按「高 beta 降、低 beta 升」重分配
    预算（等权市场的 beta 均值恒为 1，故行和天然 ≈ 1，属组合内部的风险再分配）。

四个策略均为本仓库 100% 原创实现，仅用 numpy/pandas，离线且确定性。

防未来函数：第 t 期权重只使用截至 t（含 t 收盘）的价格/收益——滚动分位/ER/z-score/
beta 均为 trailing 窗口统计；EM 为 walk-forward：只在 refit 日用**截至该日**的历史
收益重拟合，两次 refit 之间参数冻结、仅对当期收益做后验打分；预热期（统计量未知）
一律不建仓（全现金），绝不用未来数据回填。引擎还会再滞后一期，双重保险。

权重规则：long_only（≥ 0），每行和 ≤ 1（等预算 1/N 或组合层敞口缩放），
``_finalize`` 做最后一道对齐、裁剪与行内缩放兜底。
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

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
    """已实现波动率（年化）：滚动标准差 × sqrt(ppy)，样本不足则 NaN。"""
    w = max(int(window), 2)
    return ret.rolling(w, min_periods=w).std() * np.sqrt(float(periods_per_year))


def _rolling_percentile(x: pd.Series, lookback: int, min_obs: int) -> pd.Series:
    """当前值在其**过去** lookback 期历史分布中的分位（mid-rank 约定，∈[0,1]）。

    第 t 期只与 ``x[t-lookback..t-1]`` 比较（不含当期自身，避免自我参照）；
    并列取 0.5 权重（mid-rank），常数序列得 0.5 而非退化的 1.0。
    历史有效样本 < min_obs 或当期 NaN → NaN（预热期不建仓）。
    """
    lb = max(int(lookback), 2)
    mo = max(int(min_obs), 2)

    def _pct(w: np.ndarray) -> float:
        cur = w[-1]
        if not np.isfinite(cur):
            return np.nan
        past = w[:-1]
        past = past[np.isfinite(past)]
        if past.size < mo:
            return np.nan
        n_less = float(np.count_nonzero(past < cur))
        n_eq = float(np.count_nonzero(past == cur))
        return (n_less + 0.5 * n_eq) / float(past.size)

    return x.rolling(lb + 1, min_periods=mo + 1).apply(_pct, raw=True)


def _piecewise_scale(x: pd.Series, xp: Sequence[float], fp: Sequence[float]) -> pd.Series:
    """按锚点 (xp, fp) 对分位做分段线性映射（多档状态的连续缩放），NaN 保持 NaN。"""
    xv = x.to_numpy(dtype="float64")
    out = np.interp(xv, np.asarray(xp, dtype="float64"), np.asarray(fp, dtype="float64"))
    out = np.where(np.isfinite(xv), out, np.nan)
    return pd.Series(out, index=x.index)


def _hysteresis_state(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """面板双阈值滞后带状态机：enter 置 1 并保持，exit_ 清 0，其间沿用上一状态。

    进/出用不同阈值形成滞后带，显著降低状态在临界值附近的逐日抖动；
    逐行前向递推，第 t 行状态只依赖 ≤ t 的触发条件（因果、无未来）。
    """
    e = enter.fillna(False).to_numpy(dtype=bool)
    x = exit_.reindex(index=enter.index, columns=enter.columns).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(e.shape, dtype=float)
    hold = np.zeros(e.shape[1], dtype=float)
    for t in range(e.shape[0]):
        hold = np.where(x[t], 0.0, np.where(e[t], 1.0, hold))
        out[t] = hold
    return pd.DataFrame(out, index=enter.index, columns=enter.columns)


def _efficiency_ratio(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """效率比 ER = |净位移| / 路径长度 ∈ [0,1]（Kaufman efficiency ratio）。

    净位移 = |p_t − p_{t-window}|；路径长度 = Σ|p_i − p_{i-1}|（窗口内逐日绝对变动）。
    单边趋势 ER → 1，来回震荡 ER → 0。窗口内含 NaN 或路径为 0 → NaN。
    """
    L = max(int(window), 2)
    direction = (close - close.shift(L)).abs()
    path = close.diff().abs().rolling(L, min_periods=L).sum()
    return direction / path.where(path > _EPS)


# --------------------------------------------------------------------------
# 自研 2 分量一维高斯混合 EM（确定性，无随机数，log 域数值稳定）
# --------------------------------------------------------------------------
def _em_responsibility(x: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                       var: np.ndarray) -> np.ndarray:
    """E 步：返回 (n, 2) 后验责任矩阵（log-sum-exp 技巧，密度下溢也不会 0/0）。"""
    pi_c = np.clip(np.asarray(pi, dtype="float64"), 1e-12, None)
    log_pi = np.log(pi_c / pi_c.sum())
    lp = np.empty((x.size, 2), dtype="float64")
    for k in range(2):
        lp[:, k] = log_pi[k] - 0.5 * (np.log(2.0 * np.pi * var[k])
                                      + (x - mu[k]) ** 2 / var[k])
    m = lp.max(axis=1, keepdims=True)
    e = np.exp(lp - m)
    return e / e.sum(axis=1, keepdims=True)


def _em_fit_gauss2(x: np.ndarray, n_iter: int = 60,
                   tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一维 2 分量高斯混合 EM（本仓库自研，确定性）。

    初始化（无随机数）：按**中位数**把样本分成两半，两半的均值作两分量初始均值，
    总体方差作初始方差，分量权重 0.5/0.5；退化样本（近常数）用 ±sd/4 撑开。
    迭代：log 域 E 步 → M 步（方差带数据自适应下限，防分量塌缩成尖峰），
    参数变化量 < tol 或达到 n_iter 停止。返回 (pi, mu, var)，各 shape (2,)。
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = int(x.size)
    v = float(np.var(x)) if n > 0 else 0.0
    floor = max(v * 1e-6, 1e-14)
    med = float(np.median(x)) if n > 0 else 0.0
    sd = np.sqrt(max(v, 0.0))
    if n < 4:
        # 样本极少：不做真正的 EM，给出确定性退化参数
        return (np.array([0.5, 0.5]),
                np.array([med - max(sd, 1e-8), med + max(sd, 1e-8)]),
                np.array([max(v, floor), max(v, floor)]))
    lo, hi = x[x <= med], x[x > med]
    if lo.size >= 1 and hi.size >= 1:
        mu0, mu1 = float(lo.mean()), float(hi.mean())
    else:
        mu0, mu1 = med - max(sd, 1e-8), med + max(sd, 1e-8)
    if mu1 - mu0 < 1e-12:
        spread = max(sd, 1e-8) / 4.0
        mu0, mu1 = med - spread, med + spread
    pi = np.array([0.5, 0.5])
    mu = np.array([mu0, mu1])
    var = np.array([max(v, floor), max(v, floor)])
    scale = max(v, 1e-16)
    for _ in range(max(int(n_iter), 1)):
        resp = _em_responsibility(x, pi, mu, var)
        nk = resp.sum(axis=0)
        if float(nk.min()) < 1e-8:
            break                                   # 分量退化：冻结当前参数
        mu_new = (resp * x[:, None]).sum(axis=0) / nk
        d = x[:, None] - mu_new[None, :]
        var_new = np.maximum((resp * d * d).sum(axis=0) / nk, floor)
        pi_new = nk / float(n)
        delta = (abs(float(pi_new[1] - pi[1]))
                 + float(np.abs(mu_new - mu).sum()) / max(sd, 1e-8)
                 + float(np.abs(var_new - var).sum()) / scale)
        pi, mu, var = pi_new, mu_new, var_new
        if delta < float(tol):
            break
    return pi, mu, var


def _em_posterior(x: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                  var: np.ndarray, comp: int) -> np.ndarray:
    """冻结参数下，样本属于分量 ``comp`` 的后验概率（在线打分，不再拟合）。"""
    resp = _em_responsibility(np.asarray(x, dtype="float64"), pi, mu, var)
    return resp[:, int(comp)]


# --------------------------------------------------------------------------
# 权重装配小工具
# --------------------------------------------------------------------------
def _broadcast(scale: pd.Series, data: MarketData) -> pd.DataFrame:
    """组合层敞口序列 → 等权权重面板（每资产 scale/N，行和 = scale）。"""
    n = max(int(data.n_assets), 1)
    per = scale.reindex(data.dates).fillna(0.0).to_numpy(dtype="float64") / float(n)
    return pd.DataFrame(np.repeat(per[:, None], n, axis=1),
                        index=data.dates, columns=data.symbols)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、NaN → 0、裁到 [0, cap]，并把行和超 cap 的行等比缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0.0).clip(lower=0.0, upper=cap))
    total = out.sum(axis=1)
    factor = (cap / total.where(total > _EPS)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


# --------------------------------------------------------------------------
# 策略 1：多档分位波动状态择时（连续缩放）
# --------------------------------------------------------------------------
class RegimeVolTimingStrategy(Strategy):
    """波动状态择时：已实现波动的滚动分位分低/中/高三档，总敞口按分位连续缩放。"""

    name = "regime_vol_timing"
    channel = "regime"
    universe = "overlay"
    long_only = True
    description = ("多档分位波动状态择时：对等权市场组合的已实现波动计算其在过去 lookback 期"
                   "历史分布中的分位（mid-rank），以 q_low/q_mid 两个分位锚点划分低/中/高三档"
                   "状态，总敞口按分位分段线性连续缩放 1.0 → scale_mid → scale_high："
                   "低波满仓、中波按比例降、高波深度降杠杆，差额即现金（每资产 scale/N）。")
    hypothesis = ("核心假设：波动率有强聚集性（状态可短期预测），且高波动状态下单位风险收益"
                  "系统性更差（杠杆效应、被迫去杠杆、流动性收缩）。滚动分位使阈值自适应不同"
                  "时代/资产的波动水平；相对二元开关（如 vol_regime_filter），多档连续缩放把"
                  "『状态强度』映射为『仓位强度』，避免在阈值附近满仓/空仓来回跳变，换手更低、"
                  "敞口路径更平滑。失效场景：低波平静期后的突发崩盘（分位阈值滞后于跳升的"
                  "波动，减仓晚一步）、高波动伴随强反弹的市场（深度降杠杆踏空）、以及波动"
                  "水平长期漂移时分位的相对性导致『新常态』被误判为高波状态。")
    source = ("波动率 regime switching 范式的「多档分位 + 连续缩放」改写（与 volatility 渠道"
              "二元 vol_regime_filter 互补，档位映射与 mid-rank 分位为本仓库原创实现）。")
    params = {
        "vol_window": 20,       # 已实现波动窗口（日）
        "lookback": 250,        # 分位排名的历史回看窗口（只用严格早于当期的历史）
        "min_obs": 60,          # 分位所需最少历史有效样本（预热期不建仓）
        "q_low": 0.4,           # 低/中波状态分界分位：≤ q_low 满仓
        "q_mid": 0.8,           # 中/高波状态分界分位：q_low→q_mid 线性降到 scale_mid
        "scale_mid": 0.6,       # 中波档上沿（q_mid 处）的敞口
        "scale_high": 0.2,      # 极端高波（分位→1）的敞口下限
        "vol_floor": 0.01,      # 年化波动下限，避免近零波动污染分位
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        rv = _realized_vol(_market_returns(data), int(p["vol_window"]),
                           data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        pct = _rolling_percentile(rv, int(p["lookback"]), int(p["min_obs"]))
        xp = [0.0, float(p["q_low"]), float(p["q_mid"]), 1.0]
        fp = [1.0, 1.0, float(p["scale_mid"]), float(p["scale_high"])]
        scale = _piecewise_scale(pct, xp, fp).fillna(0.0)   # 分位未知（预热）→ 现金
        return _finalize(_broadcast(scale, data), data)


# --------------------------------------------------------------------------
# 策略 2：趋势/震荡状态切换（两套子逻辑按状态切换）
# --------------------------------------------------------------------------
class TrendRegimeSwitchStrategy(Strategy):
    """趋势/震荡切换：ER 状态机判别市场性格，趋势期动量持有、震荡期超跌买入。"""

    name = "trend_regime_switch"
    channel = "regime"
    universe = "timing"
    long_only = True
    description = ("逐资产趋势/震荡状态切换：用效率比 ER（净位移/路径长度，平滑后过 er_on/er_off "
                   "双阈值滞后带状态机）判别『趋势 vs 震荡』——趋势期启用动量子逻辑（过去 "
                   "mom_window 期收益为正则持有强势、为负则空仓），震荡期启用均值回归子逻辑"
                   "（z-score < -entry_z 超跌买入，回归至 z > -exit_z 上方离场），两套子逻辑"
                   "按状态硬切换，等预算 1/N。")
    hypothesis = ("核心假设（适应性市场假说）：动量与均值回归各自只在适配的市场性格下有效——"
                  "动量在震荡市被反复打脸，回归在趋势市逆势接刀；ER 度量价格运动的方向纯度，"
                  "能有效区分两种性格，按状态切换子逻辑可让每段子逻辑都工作在适配环境中。"
                  "滞后带 + ER 平滑抑制状态抖动，控制切换换手。失效场景：趋势末端的 V 型反转"
                  "（ER 滞后，状态切换晚于行情切换）、ER 长期处于两阈值之间的模糊资产、以及"
                  "只做多约束下震荡子逻辑在单边阴跌初段仍会超跌买入（接刀一段后才由趋势态接管）。")
    source = ("Kaufman 效率比 / 自适应市场假说 (MHM) 的『状态 × 子策略』切换范式实现，"
              "状态机、子逻辑组合与参数结构均为本仓库原创。")
    params = {
        "er_window": 20,     # 效率比窗口
        "er_smooth": 3,      # ER 平滑窗口（降低状态抖动）
        "er_on": 0.5,        # ER 平滑值上穿该阈值 → 进入趋势态
        "er_off": 0.25,      # ER 平滑值下破该阈值 → 回到震荡态（滞后带）
        "mom_window": 20,    # 趋势态子逻辑：动量回看窗口
        "z_window": 24,      # 震荡态子逻辑：z-score 窗口
        "entry_z": 1.0,      # 震荡态超跌买入阈值（z < -entry_z 进场）
        "exit_z": 0.0,       # 震荡态离场阈值（z > -exit_z 离场，0 = 回到均值即走）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        er = _efficiency_ratio(close, int(p["er_window"]))
        sm = max(int(p["er_smooth"]), 1)
        er_s = er.rolling(sm, min_periods=sm).mean() if sm > 1 else er
        trend = _hysteresis_state(er_s > float(p["er_on"]),
                                  er_s < float(p["er_off"]))          # 1=趋势, 0=震荡
        mom = close.pct_change(int(p["mom_window"]))
        mom_hold = ((mom > 0.0) & mom.notna()).astype(float)           # 趋势态：持有强势
        z = ind.rolling_zscore(close, int(p["z_window"]))
        mr = _hysteresis_state(z < -float(p["entry_z"]),
                               z > -float(p["exit_z"]))                # 震荡态：超跌买入
        strength = (trend * mom_hold.to_numpy(dtype="float64")
                    + (1.0 - trend) * mr.to_numpy(dtype="float64"))
        w = pd.DataFrame(strength, index=data.dates, columns=data.symbols)
        return _finalize(w / max(int(data.n_assets), 1), data)


# --------------------------------------------------------------------------
# 策略 3：自研 2 状态高斯 EM（walk-forward bull/bear 识别）
# --------------------------------------------------------------------------
class EmRegimeStrategy(Strategy):
    """自研 2 分量高斯 EM walk-forward 识别 bull/bear，后验概率连续决定总敞口。"""

    name = "em_regime"
    channel = "regime"
    universe = "overlay"
    long_only = True
    description = ("自研 2 状态高斯 EM 择时：对等权组合收益用 2 分量一维高斯混合 EM（numpy "
                   "自研、中位数分割确定性初始化、无随机数）做 walk-forward 拟合——每 refit 期"
                   "只用截至当期的历史收益重拟合一次，均值较高的分量判为 bull；当期总敞口 = "
                   "当期收益属于 bull 分量的后验概率线性映射到 [bear_expo, bull_expo]，两次 "
                   "refit 之间参数冻结、仅在线打分（bull 满仓、bear 空仓/降杠杆，每资产敞口/N）。")
    hypothesis = ("核心假设：市场收益是 bull/bear 两种状态的高斯混合（Hamilton regime "
                  "switching 的简化），EM 后验概率是比硬阈值更平滑的『状态置信度』——bear 证据"
                  "逐步累积时仓位连续下降，避免开关式择时在临界点反复抖动；walk-forward 重拟合"
                  "保证状态参数跟随分布漂移且严格无未来。失效场景：单一状态市场被强行二分"
                  "（后验在 0.5 附近漂移，仓位中庸无增益）、状态突变（下次 refit 前参数冻结，"
                  "识别滞后最多 refit 期）、以及厚尾/非高斯收益使高斯分量误判极端事件为状态切换。")
    source = ("Hamilton(1989) 马尔可夫状态切换思想的『2 分量高斯混合 EM + walk-forward』简化"
              "实现；EM 算法、确定性初始化、log 域责任计算与冻结参数在线打分均为本仓库原创，"
              "未引入任何第三方 ML 库。")
    params = {
        "min_train": 250,     # 首次拟合所需最少历史收益样本（预热期全现金）
        "refit": 21,          # walk-forward 重拟合周期（约每月一次）
        "n_iter": 60,         # EM 最大迭代次数
        "tol": 1e-7,          # EM 收敛阈值（参数相对变化量）
        "bull_expo": 1.0,     # bull 后验=1 时的总敞口（满仓）
        "bear_expo": 0.0,     # bull 后验=0 时的总敞口（全现金）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        x = _market_returns(data).to_numpy(dtype="float64")
        T = int(x.size)
        expo = np.zeros(T, dtype="float64")
        min_train = max(int(p["min_train"]), 8)
        refit = max(int(p["refit"]), 1)
        lo, hi_expo = float(p["bear_expo"]), float(p["bull_expo"])
        # walk-forward：refit 网格锚定绝对索引，训练集 = 截至 t_k 的历史收益（防未来）
        for t_k in range(min_train, T, refit):
            train = x[1:t_k + 1]                    # 跳过首行占位 0 收益
            if train.size < min_train:
                continue
            pi, mu, var = _em_fit_gauss2(train, int(p["n_iter"]), float(p["tol"]))
            bull = int(np.argmax(mu))               # 均值较高的分量 = bull（并列取 0，确定性）
            seg_hi = min(t_k + refit, T)
            p_bull = _em_posterior(x[t_k:seg_hi], pi, mu, var, bull)
            expo[t_k:seg_hi] = lo + (hi_expo - lo) * p_bull   # 参数冻结、在线打分
        scale = pd.Series(expo, index=data.dates).clip(lower=0.0, upper=1.0)
        return _finalize(_broadcast(scale, data), data)


# --------------------------------------------------------------------------
# 策略 4：滚动 beta 状态择时（组合内部风险再分配）
# --------------------------------------------------------------------------
class BetaTimingStrategy(Strategy):
    """beta 择时：滚动估计各资产相对等权市场的 beta，高 beta 降、低 beta 升。"""

    name = "beta_timing"
    channel = "regime"
    universe = "timing"
    long_only = True
    description = ("滚动 beta 状态择时：以等权组合为『市场』代理，逐资产滚动 window 期估计 "
                   "beta（cov/var，仅用历史收益），权重乘数 g = clip(1 + k×(1 − beta), g_min, "
                   "g_max)——高 beta（高风险状态）资产按比例降敞口、低 beta 资产按比例提高，"
                   "每资产基础预算 1/N；等权市场的 beta 截面均值恒为 1，故行和天然 ≈ 1，"
                   "超出 1 的行由 _finalize 等比缩回（组合层 ≤ 1，不加杠杆）。")
    hypothesis = ("核心假设：beta 是时变且有持续性的『风险状态』——资产对市场的敏感度升高时，"
                  "其下行弹性与系统性风险贡献同步升高，而预期收益补偿并不同步升高（低 beta "
                  "异象 / betting-against-beta 逻辑），因此按 1−beta 线性重分配预算可改善组合"
                  "风险调整后收益。行和恒等式（等权 beta 均值 = 1）使该策略是纯『组合内部风险"
                  "再分配』，不改变总敞口、不做择时赌博。失效场景：高 beta 资产领涨的强牛阶段"
                  "降配跑输、beta 估计窗口与真实风险状态切换错配（滞后一个窗口）、以及同质化"
                  "行情中各资产 beta 都贴近 1，乘数失去区分度。")
    source = ("低 beta 异象 / betting-against-beta 范式的『滚动 beta 预算再分配』择时改写"
              "（纯价格收益实现，cov/var 滚动估计与行和恒等式约束为本仓库原创）。")
    params = {
        "window": 60,        # beta 滚动估计窗口（日）
        "k": 1.0,            # beta 偏离的惩罚/奖励斜率：g = 1 + k×(1 − beta)
        "g_min": 0.25,       # 乘数下限（最高 beta 资产也保留 1/4 基础预算）
        "g_max": 1.75,       # 乘数上限（防止近零 beta 资产吃满预算）
        "var_eps": 1e-16,    # 市场方差下限，避免除零（低于该值视为 beta 未知 → 现金）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        W = max(int(p["window"]), 2)
        r = data.returns(1)
        mkt = r.mean(axis=1)
        # 滚动 cov/beta 用「均值差」形式向量化：cov = E[xy] − E[x]E[y]，var 同理
        mean_x = r.rolling(W, min_periods=W).mean()
        mean_y = mkt.rolling(W, min_periods=W).mean()
        mean_xy = r.mul(mkt, axis=0).rolling(W, min_periods=W).mean()
        mean_yy = mkt.pow(2).rolling(W, min_periods=W).mean()
        cov = mean_xy.sub(mean_x.mul(mean_y, axis=0))
        var = (mean_yy - mean_y.pow(2)).clip(lower=0.0)
        beta = cov.div(var.where(var > float(p["var_eps"])), axis=0)
        g = (1.0 + float(p["k"]) * (1.0 - beta)).clip(float(p["g_min"]),
                                                      float(p["g_max"]))
        g = g.where(beta.notna(), 0.0).fillna(0.0)     # beta 未知（预热）→ 现金
        w = g / float(max(int(data.n_assets), 1))
        return _finalize(w, data)


# 便于外部（报告/研究记录）按渠道枚举的辅助常量
CHANNEL = "regime"
STRATEGY_NAMES = (
    "regime_vol_timing",
    "trend_regime_switch",
    "em_regime",
    "beta_timing",
)

__all__ = [
    "RegimeVolTimingStrategy",
    "TrendRegimeSwitchStrategy",
    "EmRegimeStrategy",
    "BetaTimingStrategy",
    "CHANNEL",
    "STRATEGY_NAMES",
]
