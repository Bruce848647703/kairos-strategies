"""渠道：pairs —— 配对交易（pairs trading）家族的三个变体。

本渠道只做「两腿反向、逐行权重之和 ≈ 0」的市场中性配对组合，与已有渠道的区别是
**配对关系的来源不同**（价格距离 / 协整证据 / 结构性板块），而不是信号形式不同：

1. ``ssd_pairs``             —— 距离法（SSD, Sum of Squared Difference，Gatev-Goetzmann-
   Rouwenhorst 范式的自研实现）：在形成期窗口内把每只价格除以窗口首日价格做**归一化**，
   两两计算归一化路径的平方差之和 SSD，取「历史走势最像」的一对；交易期用该对的
   归一化价差（price-relative spread）滚动 z-score，价差被高估则做空贵腿/做多便宜腿。
   与 statarb 渠道的 ``coint_pairs``（协整门槛 + 估出的 β + 残差）不同：此处**不做任何
   平稳性检验、不估 β**，纯粹按价格路径的几何距离选对（β 隐含为 1，作用于归一化价格）。
2. ``coint_pairs_portfolio`` —— 协整配对**组合**：Engle-Granger 两步法的第一步用 numpy
   OLS 估对冲比率 β，第二步对残差同时做两个自研平稳性诊断——(a) 带 ``adf_lags`` 阶滞后
   差分项的 ADF 风格单位根回归（t 统计量由 ``(X'X)⁻¹σ²`` 通式求得，不是单变量解析式）；
   (b) 方差比 ``VR(q) = Var_q(Δe) / (q·Var_1(Δe))``（随机游走 ≈ 1、均值回归 < 1）。
   两道门槛都通过的资产对**全部入选**（按证据强度排序、腿不重复地贪心取前 ``max_pairs``
   对），预算等分到每一对，各自按残差 z-score 反向持仓。与 statarb 的 ``coint_pairs``
   的区别：那个只取**一对**最优、诊断只有无滞后 ADF + 半衰期、排序键以 R² 优先；这里是
   **多对分散**、双重诊断（ADF + 方差比）、排序以协整证据强度优先。
3. ``sector_neutral_pairs``  —— 板块中性配对：先用尾部收益相关系数矩阵构造距离
   ``d = 1 - corr``，跑一个**自研平均链接层次聚类**（numpy，无 scipy）把资产并成
   ``n_sectors`` 个「板块」；再在每个板块**内部**按相关性最强的顺序取出互不共用腿的配对，
   用 OLS β 的对数残差 z-score 反向持仓。因为每一对都在板块内部且两腿等预算反向，
   所以**每个板块的净敞口恒为 0**（板块中性），整体行和也恒为 0。

统一后处理 ``_dollar_neutral``：逐行在**有权重的资产**内去均值（把行和压到 ≈ 0），
再按「每行绝对值之和 ≤ budget(=1)」整体等比缩放（不加杠杆），最后对齐 index/columns。
配对组合本身已是「±同幅两腿」结构，去均值通常是无操作（行和本来就是 0），该函数只是
数值上的最后一道保险，因此**行和 ≈ 0 与毛敞口 ≤ 1 是硬约束**。

防未来函数：所有估计（SSD 距离、OLS β、ADF/方差比、相关矩阵与聚类）都发生在
「锚定样本起点的再平衡网格」上，且只用**尾部窗口**；同一 block 内系数与配对冻结，
t 期权重只依赖 ≤ t 的价格。网格锚定在绝对索引 0（而非样本末尾）是「篡改/追加未来
数据不会改写历史权重」的关键（见 ``tests/test_pairs.py`` 的 tamper 测试）。
全程离线、无随机数、确定性；协整/β/平稳性/聚类全部为 numpy 自研，不用
statsmodels / sklearn / scipy。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-9           # 价格下限保护，避免 log(0) / 除零
_TINY = 1e-12         # 退化判定阈值（方差为 0、完全共线等）
_MIN_OBS = 24         # 任何统计诊断所需的最少样本数

PairSpec = Tuple[Tuple[int, int], float, float]   # ((x 腿索引, y 腿索引), alpha, beta)


# ------------------------------------------------------------------ 通用工具

def _price_panel(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64、裁到正数、前向填充（只用过去，不看未来）。"""
    p = data.prices.astype("float64").clip(lower=_EPS)
    p = p.ffill()                       # 停牌/缺失：沿用上一期价格（不含未来信息）
    return p.fillna(1.0)                # 开头就缺失的极端情况：退化为常数 1.0


def _log_panel(data: MarketData) -> np.ndarray:
    """对数价格矩阵 (T, N)，float64。"""
    return np.log(_price_panel(data).to_numpy(dtype="float64"))


def _rebalance_grid(n_rows: int, start: int, freq: int) -> List[int]:
    """再平衡索引网格 ``start, start + freq, ...``（**锚定样本起点**）。

    锚定在绝对索引 0 而不是样本末尾：追加/篡改未来数据既不会移动历史 block 的边界，
    也不会改变任何一次再平衡的内容 —— 这是防未来函数（前缀/篡改不变性）的结构性保证。
    """
    start = int(start)
    if start < 0 or start >= int(n_rows):
        return []
    return list(range(start, int(n_rows), max(int(freq), 1)))


def _signal_ramp(z: np.ndarray, exit_z: float, entry_z: float) -> np.ndarray:
    """死区 + 线性斜坡的连续仓位强度 ∈ [-1, 1]（滞后带的连续版本，压低换手）。

    ``|z| <= exit_z`` -> 0（中枢附近不交易）；``|z| >= entry_z`` -> ±1（满强度）；
    两者之间用 ``np.interp`` 线性插值，故 z 在阈值附近抖动时权重连续变化而非跳变。
    非有限值（预热期 NaN、退化窗口）一律当作 0 —— **不确定就不下注**。
    """
    z = np.asarray(z, dtype="float64")
    z = np.where(np.isfinite(z), z, 0.0)
    lo, hi = float(exit_z), float(entry_z)
    if hi <= lo:
        mag = (np.abs(z) >= hi).astype("float64")     # 退化为硬阈值
    else:
        mag = np.interp(np.abs(z), [lo, hi], [0.0, 1.0])
    return np.sign(z) * mag


def _rolling_z(values: np.ndarray, index: pd.Index, window: int) -> np.ndarray:
    """对一维序列做尾部窗口滚动 z-score（``ind.rolling_zscore``），返回 float64 数组。

    只使用 ``[t - window + 1, t]`` 的数据，因此逐行都是「截至当期」的标准化偏离；
    预热期与零方差窗口返回 NaN，由 ``_signal_ramp`` 折成 0（不交易）。
    """
    s = pd.Series(np.asarray(values, dtype="float64"), index=index)
    z = ind.rolling_zscore(s, int(window))
    return np.asarray(z.to_numpy(dtype="float64"), dtype="float64")


def _pair_candidates(n: int) -> List[Tuple[int, int]]:
    """全部无序资产对 ``(i, j), i < j``，按字典序排列（确定性、可复现）。"""
    n = int(n)
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _dollar_neutral(raw: pd.DataFrame, data: MarketData, budget: float = 1.0) -> pd.DataFrame:
    """把原始信号后处理成「美元中性 + 不加杠杆」的目标权重面板。

    逐行（逐日）在**有权重的资产**（原始信号有限且非零者）上：
      1. 去均值 -> 行和严格压到 ≈ 0（美元/市场中性）；
      2. 若绝对值之和 > ``budget`` 则整行等比缩放 -> 每行绝对值和 ≤ budget（无杠杆）；
      3. 有效资产不足 2 个（凑不出多空两腿）时整行置 0。
    输出对齐 ``data`` 的 index/columns，不含 NaN。
    """
    aligned = (raw.reindex(index=data.dates, columns=data.symbols)
                  .apply(pd.to_numeric, errors="coerce"))
    v = np.array(aligned.to_numpy(dtype="float64"), copy=True)
    v[~np.isfinite(v)] = 0.0
    active = v != 0.0
    cnt = active.sum(axis=1)
    offset = np.where(cnt > 0, v.sum(axis=1) / np.maximum(cnt, 1), 0.0)
    cen = np.where(active, v - offset[:, None], 0.0)
    gross = np.abs(cen).sum(axis=1)
    scale = np.where(gross > budget, budget / np.where(gross > _TINY, gross, 1.0), 1.0)
    out = cen * scale[:, None]
    out[cnt < 2] = 0.0
    out[~np.isfinite(out)] = 0.0
    return pd.DataFrame(out, index=data.dates, columns=data.symbols)


def _ols_ab(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """一元 OLS ``y = alpha + beta * x`` -> (alpha, beta)（lstsq，确定性）。"""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    design = np.column_stack([np.ones(x.size, dtype="float64"), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _ols_t_stats(design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """通用多元 OLS：返回 ``(coef, t 统计量向量)``，纯 numpy（不用 statsmodels）。

    ``Var(coef) = σ² · (X'X)⁻¹``，``σ² = RSS / (n - k)``；``t = coef / se``。
    设计矩阵退化（``X'X`` 不满秩）时用 ``pinv`` 兜底，对应系数的 t 记 0（证据不足）。
    """
    X = np.asarray(design, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = float(resid @ resid) / float(max(n - k, 1))
    cov = np.linalg.pinv(X.T @ X) * sigma2
    var = np.clip(np.diag(cov), 0.0, None)
    se = np.sqrt(var)
    t = np.where(se > _TINY, coef / np.where(se > _TINY, se, 1.0), 0.0)
    return coef, np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _adf_diagnostics(resid: np.ndarray, lags: int = 2) -> Tuple[float, float, float]:
    """残差单位根回归（Engle-Granger 第二步的自研 numpy 版，**带滞后差分项**）。

    对残差跑 ADF 风格回归
    ``Δe_t = c + λ·e_{t-1} + Σ_{m=1..lags} φ_m·Δe_{t-m} + ε_t``，
    用 ``_ols_t_stats`` 的通式（``(X'X)⁻¹σ²``）取 ``λ`` 的 t 统计量。返回
    ``(adf_t, rho, half_life)``：

    * ``adf_t``：越负表示拒绝「残差有单位根」（即拒绝不协整）的证据越强；本实现
      **不查 Dickey-Fuller 临界值表、不给 p 值**，只用于同序排序 + 宽松地板阈值；
    * ``rho = 1 + λ``：残差 AR(1) 系数，``rho < 1`` 即均值回归；
    * ``half_life = -ln2 / ln(rho)``：回归一半所需期数（``rho >= 1`` 记 ``inf``，
      ``rho <= 0`` 的过度回归/振荡记 1.0）。

    样本不足或退化时返回 ``(0.0, 1.0, inf)``（视为「没有协整证据」）。
    """
    e = np.asarray(resid, dtype="float64")
    lags = int(max(lags, 0))
    de = np.diff(e)
    rows = de.size - lags
    if rows < _MIN_OBS or not np.isfinite(e).all():
        return 0.0, 1.0, np.inf
    dep = de[lags:]                                  # Δe_t,      t = lags+1 .. T-1
    level = e[lags:-1]                               # e_{t-1}
    cols = [np.ones(rows, dtype="float64"), level]
    for m in range(1, lags + 1):                     # Δe_{t-m}
        cols.append(de[lags - m: de.size - m])
    coef, t = _ols_t_stats(np.column_stack(cols), dep)
    lam = float(coef[1])
    rho = 1.0 + lam
    if not np.isfinite(rho) or rho >= 1.0 - 1e-12:
        return float(t[1]), rho, np.inf
    if rho <= 0.0:
        return float(t[1]), rho, 1.0
    return float(t[1]), rho, float(-np.log(2.0) / np.log(rho))


def _variance_ratio(e: np.ndarray, lag: int = 5) -> float:
    """方差比 ``VR(q) = Var(e_t - e_{t-q}) / (q · Var(e_t - e_{t-1}))``（自研，无 scipy）。

    随机游走的 q 期方差恰好是 1 期方差的 q 倍 -> ``VR ≈ 1``；均值回归序列的 q 期
    方差增长慢于线性 -> ``VR < 1``；趋势/爆炸序列 -> ``VR > 1``。这是与 ADF 回归
    **相互独立**的第二道平稳性证据：ADF 看的是「回归系数是否显著为负」，方差比看的是
    「长期波动是否被均值回归压住」，两者同时通过才算协整对（降低小样本伪回归误判）。
    数据不足或退化返回 ``inf``（视为不平稳）。
    """
    e = np.asarray(e, dtype="float64")
    q = int(max(lag, 1))
    if e.size < q + _MIN_OBS or not np.isfinite(e).all():
        return np.inf
    d1 = np.diff(e)
    v1 = float(np.var(d1, ddof=1))
    if v1 <= _TINY:
        return np.inf
    dq = e[q:] - e[:-q]
    if dq.size < _MIN_OBS:
        return np.inf
    return float(np.var(dq, ddof=1) / (q * v1))


def _ssd_matrix(norm_win: np.ndarray) -> np.ndarray:
    """归一化价格窗口 -> 成对 SSD（平方差之和）距离矩阵（对称、对角为 0）。

    用恒等式 ``SSD(i,j) = Σa_i² + Σa_j² - 2·Σa_i·a_j`` 一次矩阵乘法算完全部对，
    再夹到非负（消除浮点噪声造成的微小负值）。
    """
    A = np.asarray(norm_win, dtype="float64")
    sq = (A ** 2).sum(axis=0)
    d = sq[:, None] + sq[None, :] - 2.0 * (A.T @ A)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def _corr_matrix(returns_win: np.ndarray) -> np.ndarray:
    """尾部收益窗口 -> 相关系数矩阵（纯 numpy；退化列的相关系数记 0，对角为 1）。"""
    X = np.asarray(returns_win, dtype="float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - X.mean(axis=0)
    sd = np.sqrt((Xc ** 2).sum(axis=0))
    good = sd > _TINY
    denom = np.where(good, sd, 1.0)
    C = (Xc.T @ Xc) / np.outer(denom, denom)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C[~good, :] = 0.0
    C[:, ~good] = 0.0
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def _agglomerative_groups(dist: np.ndarray, k: int) -> List[np.ndarray]:
    """平均链接（average-linkage）层次聚类，把 n 个资产并成 k 组（自研 numpy 版）。

    每轮取当前**组间平均距离最小**的两组合并（距离按组大小加权平均更新），直到只剩 k 组。
    并列时用 ``np.triu_indices`` 的行主序 + ``argmin`` 取索引字典序最小的一对，
    因此结果与调用次数无关（确定性）。返回按「组内最小成员索引」升序的分组列表，
    分组是全体资产的一个划分（无遗漏、无重叠）。
    """
    D = np.array(dist, dtype="float64", copy=True)
    D[~np.isfinite(D)] = 2.0
    n = int(D.shape[0])
    np.fill_diagonal(D, np.inf)
    k = int(np.clip(k, 1, max(n, 1)))
    members: List[List[int]] = [[i] for i in range(n)]
    while len(members) > k:
        m = len(members)
        iu = np.triu_indices(m, 1)
        vals = D[iu]
        if vals.size == 0 or not np.isfinite(vals).any():
            break                                   # 无可合并信息，保持现状
        pos = int(np.argmin(vals))                  # 首个最小值 -> 字典序最小的一对
        a, b = int(iu[0][pos]), int(iu[1][pos])
        na, nb = float(len(members[a])), float(len(members[b]))
        merged = (D[a] * na + D[b] * nb) / (na + nb)     # 平均链接更新
        keep = [x for x in range(m) if x != a and x != b]
        nxt = np.full((len(keep) + 1, len(keep) + 1), np.inf, dtype="float64")
        nxt[:len(keep), :len(keep)] = D[np.ix_(keep, keep)]
        nxt[:len(keep), len(keep)] = merged[keep]
        nxt[len(keep), :len(keep)] = merged[keep]
        D = nxt
        members = [members[x] for x in keep] + [sorted(members[a] + members[b])]
    return [np.array(sorted(g), dtype=int) for g in sorted(members, key=lambda g: g[0])]


# ------------------------------------------------------------------ 策略 1：SSD 距离法配对

class SsdPairsStrategy(Strategy):
    name = "ssd_pairs"
    channel = "pairs"
    universe = "cross_section"
    long_only = False
    description = ("距离法（SSD）配对交易：在形成期窗口内把每只价格除以窗口首日价格做归一化，"
                   "用一次矩阵乘法算出全部资产对的归一化路径平方差之和 SSD，取「历史走势最像」"
                   "的一对（SSD 最小，并列按资产索引字典序）；交易期沿用同一归一化基准构造价差 "
                   "norm_y - norm_x，对其做尾部滚动 z-score：价差被高估(z 高)则做空 y 腿/做多 x 腿，"
                   "被低估则反向，两腿等预算反向，组合逐行权重和恒为 0。每 refit_freq 期重新选对。")
    hypothesis = ("核心假设：历史上价格走势最贴近的两只资产，往往受同一基本面/资金纽带约束，"
                  "短期背离多为流动性冲击造成的临时错价，会向「历史最像」的关系收敛，故按价差 "
                  "z-score 反向持仓可获得收敛收益，且两腿对冲掉市场 beta。失效场景：SSD 只度量"
                  "过去的**几何相似度**、不检验价差的平稳性，可能选中两只同涨同跌的趋势资产"
                  "（相似但都非平稳），此时价差发散；相似度排名频繁换位会带来换手，且窗口内发生"
                  "结构性变化（并购、政策）时形成期本身已失效。")
    source = ("配对交易距离法（distance approach / Sum of Squared Difference）经典范式——"
              "Gatev, Goetzmann & Rouwenhorst (2006) 的自研 numpy 实现：归一化价格、SSD 矩阵、"
              "最接近对筛选与价差 z-score 全部本仓库原创。与 statarb 渠道 coint_pairs（协整门槛 + "
              "估 β + ADF/半衰期诊断，只取统计上最显著的一对）、meanrev 渠道 pairs_spread（固定两腿、"
              "β=1 的对数价差 + 三态状态机）均不同：此处**不做任何平稳性检验、不估 β**，纯按价格路径"
              "距离选对，且用死区+斜坡的连续仓位而非离散三态。")
    params = {"formation_window": 90,   # 形成期：SSD 距离与归一化基准的尾部窗口
              "z_window": 30,           # 价差滚动 z-score 窗口（自动裁到 ≤ formation_window）
              "refit_freq": 10,         # 重新选对的间隔（block 内配对与基准冻结）
              "entry_z": 1.5,           # 满强度阈值
              "exit_z": 0.5,            # 死区阈值（|z| 小于它不交易）
              "leg_budget": 0.5}        # 单腿预算：两腿绝对值之和 = |仓位强度| ≤ 1


    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero                                       # 单资产无法配对
        P = _price_panel(data).to_numpy(dtype="float64")
        T = int(P.shape[0])
        fwin = max(int(self.params["formation_window"]), 2)
        zwin = int(np.clip(int(self.params["z_window"]), 2, fwin))
        start = fwin - 1
        if T <= start:
            return zero                                       # 历史不足，整段空仓
        leg = float(self.params["leg_budget"])
        entry = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        out = np.zeros((T, n), dtype="float64")
        grid = _rebalance_grid(T, start, int(self.params["refit_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            lo = t0 - fwin + 1                                # 形成期首日 = 归一化基准日
            base = P[lo]
            if not np.isfinite(base).all() or float(base.min()) <= _TINY:
                continue
            norm = P[lo:t0 + 1] / base[None, :]               # 归一化价格路径（形成期）
            pair = self._closest_pair(norm)
            if pair is None:
                continue                                      # 全部退化 -> 本 block 空仓
            i, j = pair
            # 价差沿用同一基准延伸到 block 内；z 只看尾部 zwin 期 -> 逐行只用 ≤ t 的信息
            spread = P[lo:, j] / base[j] - P[lo:, i] / base[i]
            z = _rolling_z(spread, data.dates[lo:], zwin)
            # q > 0 = 做多价差（多 j 空 i）；z 高（j 相对贵）-> q < 0 = 做空价差
            q = -_signal_ramp(z[t0 - lo:t1 - lo], exit_z, entry)
            out[t0:t1, j] = leg * q
            out[t0:t1, i] = -leg * q
        return _dollar_neutral(pd.DataFrame(out, index=data.dates, columns=data.symbols), data, 1.0)

    @staticmethod
    def _closest_pair(norm_win: np.ndarray) -> Optional[Tuple[int, int]]:
        """在归一化价格窗口上挑出 SSD 最小的资产对；全退化返回 None。

        退化（窗口内为常数、含非有限值）的资产不参与配对；排序键 ``(ssd, i, j)``
        保证并列时结果唯一。
        """
        A = np.asarray(norm_win, dtype="float64")
        if A.ndim != 2 or A.shape[1] < 2 or not np.isfinite(A).all():
            return None
        ok = A.std(axis=0) > _TINY
        if int(ok.sum()) < 2:
            return None
        d = _ssd_matrix(A)
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[float, int, int]] = None
        for i, j in _pair_candidates(A.shape[1]):
            if not (ok[i] and ok[j]):
                continue
            key = (float(d[i, j]), int(i), int(j))
            if best_key is None or key < best_key:
                best_key, best = key, (int(i), int(j))
        return best


# ------------------------------------------------------------------ 策略 2：协整配对组合

class CointPairsPortfolioStrategy(Strategy):
    name = "coint_pairs_portfolio"
    channel = "pairs"
    universe = "cross_section"
    long_only = False
    description = ("协整配对**组合**（Engle-Granger 两步法的自研 numpy 实现）：在再平衡网格的尾部窗口"
                   "上遍历全部资产对，第一步 OLS 估对冲比率 β（logy = α + β·logx），第二步对残差同时做"
                   "两道平稳性诊断——带滞后差分项的 ADF 风格单位根回归（t 统计量）与方差比 VR(q)；"
                   "两道门槛都通过的对**全部入选**（按协整证据强度排序、腿不重复地贪心取前 max_pairs 对），"
                   "把 gross_budget 等分到每一对，各对按残差 z-score 反向持仓（z 高做空贵腿/做多便宜腿）。"
                   "每对内部两腿等预算反向，故整体逐行权重和恒为 0、绝对值和 ≤ 1。")
    hypothesis = ("核心假设：真协整的资产对存在平稳的线性组合（相对估值），偏离中枢后由套利/基本面纽带"
                  "拉回；把预算分散到**多对**而非单对，可摊薄「某一对协整关系突然破裂」的特质风险，"
                  "同时多空对冲剥离市场 beta，收益来源是多个价差的收敛。失效场景：多重检验下总有一些"
                  "伪回归残差看起来平稳（尤其小样本），入选的对可能根本不协整；系统性压力期（流动性危机）"
                  "所有价差同时走阔且不收敛，分散化在最需要时失效；β 在 block 内冻结，关系漂移会带来"
                  "实现偏差。")
    source = ("统计套利经典范式——协整配对组合（cointegrated pairs portfolio / Engle-Granger）。"
              "OLS β、带滞后项的 ADF 风格 t 统计量、方差比诊断、贪心不重复选对与预算等分全部为本仓库"
              "原创 numpy 实现，不用 statsmodels / sklearn / scipy（也不查 Dickey-Fuller 临界值表、"
              "不给 p 值，只做同序排序 + 宽松地板阈值）。与 statarb 渠道 coint_pairs 的区别：那是"
              "**单对**最优（无滞后 ADF + 半衰期门槛、以 R² 优先排序、单一 leg_budget），此处是"
              "**多对分散**（ADF + 方差比双重门槛、以协整证据强度优先排序、预算等分到每对）。")
    params = {"est_window": 120,       # 协整回归 / 平稳性诊断的尾部窗口
              "z_window": 30,          # 残差滚动 z-score 窗口（自动裁到 ≤ est_window）
              "refit_freq": 20,        # 重新选对/重估 β 的间隔（block 内冻结）
              "entry_z": 1.5,          # 满强度阈值
              "exit_z": 0.5,           # 死区阈值
              "max_pairs": 3,          # 组合内最多持有多少对
              "adf_lags": 2,           # ADF 回归的滞后差分阶数
              "min_adf_t": -2.5,       # ADF t 统计量上限（越负越显著；宽松地板）
              "max_half_life": 60.0,   # 半衰期上限：回归太慢的价差没有交易价值
              "vr_lag": 5,             # 方差比的滞后阶 q
              "max_var_ratio": 0.75,   # 方差比上限（随机游走 ≈ 1，须显著低于 1）
              "exclusive_legs": True,  # True = 一只资产最多出现在一对里（避免重复计敞口）
              "gross_budget": 1.0}     # 组合毛敞口上限（每行绝对值和），等分到各对

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero
        L = _log_panel(data)
        T = int(L.shape[0])
        est = max(int(self.params["est_window"]), _MIN_OBS + int(self.params["adf_lags"]) + 2)
        zwin = int(np.clip(int(self.params["z_window"]), 2, est))
        start = est - 1
        if T <= start:
            return zero
        entry = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        budget = float(self.params["gross_budget"])
        out = np.zeros((T, n), dtype="float64")
        grid = _rebalance_grid(T, start, int(self.params["refit_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            chosen = self._select_pairs(L[t0 - est + 1:t0 + 1])
            if not chosen:
                continue                                      # 本 block 无合格协整对 -> 空仓
            per_pair = budget / float(len(chosen))             # 预算等分到每一对
            for (i, j), alpha, beta in chosen:
                # 全历史用同一组 (α, β) 算残差（度量单位一致），滚动 z 只看尾部 -> 逐行只用 ≤ t
                resid = L[:, j] - alpha - beta * L[:, i]
                z = _rolling_z(resid, data.dates, zwin)
                q = -_signal_ramp(z[t0:t1], exit_z, entry)     # z 高（y 贵）-> q < 0 = 做空价差
                out[t0:t1, j] += (per_pair / 2.0) * q
                out[t0:t1, i] += -(per_pair / 2.0) * q
        return _dollar_neutral(pd.DataFrame(out, index=data.dates, columns=data.symbols), data, budget)

    def _select_pairs(self, Lwin: np.ndarray) -> List[PairSpec]:
        """在尾部对数价格窗口上，挑出**多对**通过协整门槛的资产（按证据强度排序）。

        单对入选门槛（全部满足，否则跳过）：
          1. 样本充足、两腿非常数、残差非退化（不完全共线）；
          2. ``adf_t <= min_adf_t``：ADF 风格 t 统计量足够负（拒绝单位根的证据）；
          3. ``half_life <= max_half_life``：回归速度够快，偏离才有交易价值；
          4. ``VR(vr_lag) <= max_var_ratio``：长期方差增长显著慢于线性（第二道独立证据）。
        排序键 ``(adf_t, vr, -r2, half_life, i, j)`` 升序：协整证据最强者优先，其次方差比
        最低、价差最紧（r2 最高）、回归最快，最后按索引字典序 —— 并列时结果唯一。
        ``exclusive_legs=True`` 时贪心跳过与已选对共用腿的候选，保证每只资产最多一对，
        预算等分后不会出现同一资产的敞口叠加。全部不合格返回空列表 -> 该 block 空仓。
        """
        Lwin = np.asarray(Lwin, dtype="float64")
        if Lwin.ndim != 2 or Lwin.shape[1] < 2 or not np.isfinite(Lwin).all():
            return []
        n = int(Lwin.shape[1])
        lags = int(max(self.params["adf_lags"], 0))
        q = int(max(self.params["vr_lag"], 1))
        min_t = float(self.params["min_adf_t"])
        max_hl = float(self.params["max_half_life"])
        max_vr = float(self.params["max_var_ratio"])
        exclusive = bool(self.params["exclusive_legs"])
        cap = int(max(self.params["max_pairs"], 1))
        cands: List[Tuple[float, float, float, float, int, int, float, float]] = []
        for i, j in _pair_candidates(n):
            x, y = Lwin[:, i], Lwin[:, j]
            if x.size < _MIN_OBS + lags + 2:
                continue
            sxx = float(((x - x.mean()) ** 2).sum())
            syy = float(((y - y.mean()) ** 2).sum())
            if sxx <= _TINY or syy <= _TINY:
                continue                                      # 常数列：无信息
            alpha, beta = _ols_ab(x, y)
            resid = y - alpha - beta * x
            if float(resid.std()) <= _TINY:
                continue                                      # 完全共线，无残差可交易
            adf_t, _, hl = _adf_diagnostics(resid, lags)
            vr = _variance_ratio(resid, q)
            if not np.isfinite(adf_t) or adf_t > min_t:
                continue                                      # 拒绝单位根的证据不足
            if not np.isfinite(hl) or hl > max_hl:
                continue                                      # 单位根 / 回归太慢
            if not np.isfinite(vr) or vr > max_vr:
                continue                                      # 长期方差未按均值回归收敛
            r2 = 1.0 - float(resid @ resid) / syy              # 价差质量（只用于排序）
            cands.append((adf_t, vr, -r2, hl, int(i), int(j), alpha, beta))
        cands.sort(key=lambda c: (c[0], c[1], c[2], c[3], c[4], c[5]))
        chosen: List[PairSpec] = []
        used: set = set()
        for c in cands:
            i, j = int(c[4]), int(c[5])
            if exclusive and (i in used or j in used):
                continue
            chosen.append(((i, j), float(c[6]), float(c[7])))
            used.update({i, j})
            if len(chosen) >= cap:
                break
        return chosen


# ------------------------------------------------------------------ 策略 3：板块中性配对

class SectorNeutralPairsStrategy(Strategy):
    name = "sector_neutral_pairs"
    channel = "pairs"
    universe = "cross_section"
    long_only = False
    description = ("板块中性配对：先用尾部收益的相关系数矩阵构造距离 d = 1 - corr，跑自研的平均链接"
                   "层次聚类把资产并成 n_sectors 个「板块」；再在每个板块**内部**按尾部相关性最强的"
                   "顺序取出互不共用腿的配对（每块最多 max_pairs_per_sector 对），用 OLS 估 β 后对"
                   "对数残差做滚动 z-score 反向持仓。预算先等分到各活跃板块、再等分到块内各对，"
                   "每一对两腿等幅反向，故**每个板块的净敞口恒为 0（板块中性）**、整体行和恒为 0、"
                   "绝对值和 ≤ 1。")
    hypothesis = ("核心假设：相关性最高的资产对更可能同属一个「板块」（同行业/同风格/同资金链），"
                  "板块内部的相对估值偏离是流动性与情绪造成的临时错价，会收敛；把配对限制在板块内部"
                  "并让每块净敞口为 0，可同时剥离市场 beta **与板块 beta**，只承担「块内相对价值」"
                  "这一维风险，比全 universe 自由配对更抗行业性冲击。失效场景：板块划分依赖历史相关性，"
                  "相关性结构切换（板块轮动、危机期相关性齐涨）时分组不稳定、换手上升；块内资产太少"
                  "（< 2）则该块无法配对、预算浪费；若真正的错价发生在**跨板块**之间，本策略结构上"
                  "就抓不到。")
    source = ("统计套利经典范式——行业内配对 / 板块中性多空（sector-neutral pairs trading）。"
              "本仓库原创 numpy 实现：相关距离 + **平均链接层次聚类**（不用 scipy.cluster）、"
              "块内相关性排序选对、两级等分预算。与 statarb 渠道 basket_neutral（按波动分成两个篮子、"
              "交易**篮子之间**的便宜度价差）方向相反：那里是篮子间多空，这里是**篮子内部**配对、"
              "篮子之间零敞口；与 coint_pairs / ssd_pairs 的自由配对也不同，配对候选被板块结构约束。")
    params = {"corr_window": 60,          # 板块划分用的尾部收益相关窗口
              "est_window": 60,           # 块内配对的 OLS β 估计窗口
              "z_window": 25,             # 残差滚动 z-score 窗口（自动裁到 ≤ est_window）
              "refit_freq": 15,           # 重新聚类/选对的间隔（block 内冻结）
              "entry_z": 1.2,             # 满强度阈值
              "exit_z": 0.3,              # 死区阈值
              "n_sectors": 3,             # 目标板块数（自动裁到 [1, N//2]，保证块内可配对）
              "max_pairs_per_sector": 1,  # 每个板块内部最多交易几对
              "gross_budget": 1.0}        # 组合毛敞口上限（每行绝对值和）

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero
        L = _log_panel(data)
        T = int(L.shape[0])
        corr_win = max(int(self.params["corr_window"]), _MIN_OBS)
        est = max(int(self.params["est_window"]), _MIN_OBS)
        zwin = int(np.clip(int(self.params["z_window"]), 2, est))
        start = max(corr_win, est)                # 保证相关窗口不含 r[0] 的占位 0
        if T <= start:
            return zero
        r = np.zeros((T, n), dtype="float64")
        r[1:] = L[1:] - L[:-1]                    # 对数收益（首行占位 0，非未来信息）
        entry = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        budget = float(self.params["gross_budget"])
        out = np.zeros((T, n), dtype="float64")
        grid = _rebalance_grid(T, start, int(self.params["refit_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            groups = self._groups_at(L, r, t0, corr_win, n)
            active = [g for g in groups if g.size >= 2]
            if not active:
                continue
            per_sector = budget / float(len(active))          # 第一级：预算等分到活跃板块
            for g in active:
                C = _corr_matrix(r[t0 - corr_win + 1:t0 + 1][:, g])
                picks = self._pairs_in_sector(g, C, int(self.params["max_pairs_per_sector"]))
                if not picks:
                    continue
                per_pair = per_sector / float(len(picks))     # 第二级：板块内等分到各对
                for (i, j) in picks:
                    alpha, beta = _ols_ab(L[t0 - est + 1:t0 + 1, i], L[t0 - est + 1:t0 + 1, j])
                    resid = L[:, j] - alpha - beta * L[:, i]
                    z = _rolling_z(resid, data.dates, zwin)
                    q = -_signal_ramp(z[t0:t1], exit_z, entry)
                    out[t0:t1, j] += (per_pair / 2.0) * q
                    out[t0:t1, i] += -(per_pair / 2.0) * q
        return _dollar_neutral(pd.DataFrame(out, index=data.dates, columns=data.symbols), data, budget)

    def _groups_at(self, L: np.ndarray, r: np.ndarray, t0: int, corr_win: int,
                   n: int) -> List[np.ndarray]:
        """第 t0 期所属 block 的板块划分（只用 ≤ t0 的收益）。"""
        C = _corr_matrix(r[t0 - corr_win + 1:t0 + 1])
        dist = np.clip(1.0 - C, 0.0, 2.0)
        k = int(np.clip(int(self.params["n_sectors"]), 1, max(n // 2, 1)))
        return _agglomerative_groups(dist, k)

    @staticmethod
    def _pairs_in_sector(group: np.ndarray, corr: np.ndarray,
                         max_pairs: int) -> List[Tuple[int, int]]:
        """在一个板块内部按「相关性最强优先」取出互不共用腿的配对（返回**全局**资产索引）。

        ``corr`` 是该板块成员的相关子矩阵（行列顺序与 ``group`` 一致）；排序键
        ``(-corr, a, b)`` 保证并列时唯一。成员不足 2 个返回空列表。
        """
        g = np.asarray(group, dtype=int)
        cap = int(max(max_pairs, 0))
        if g.size < 2 or cap == 0:
            return []
        C = np.asarray(corr, dtype="float64")
        ranked: List[Tuple[float, int, int]] = []
        for a in range(g.size):
            for b in range(a + 1, g.size):
                c = C[a, b] if np.isfinite(C[a, b]) else -1.0
                ranked.append((-float(c), a, b))
        ranked.sort()
        picks: List[Tuple[int, int]] = []
        used: set = set()
        for _, a, b in ranked:
            if a in used or b in used:
                continue
            used.update({a, b})
            picks.append((int(g[a]), int(g[b])))
            if len(picks) >= cap:
                break
        return picks

    def sector_labels(self, data: MarketData, at: Optional[int] = None) -> np.ndarray:
        """诊断用：返回某期所属 block 的板块标签（长度 = n_assets，预热期为 -1）。

        ``at`` 是绝对行号（默认最后一行）；只用 ≤ at 的数据，故可安全用于研究核对与测试
        （例如验证「每个板块的净敞口 ≈ 0」）。标签值 = 板块在 ``_agglomerative_groups``
        返回列表中的序号（按组内最小资产索引升序）。
        """
        n = int(data.n_assets)
        T = int(len(data.dates))
        labels = np.full(n, -1, dtype=int)
        if n < 2 or T == 0:
            return labels
        corr_win = max(int(self.params["corr_window"]), _MIN_OBS)
        est = max(int(self.params["est_window"]), _MIN_OBS)
        start = max(corr_win, est)
        pos = int(T - 1 if at is None else np.clip(int(at), 0, T - 1))
        grid = _rebalance_grid(T, start, int(self.params["refit_freq"]))
        block = [t0 for t0 in grid if t0 <= pos]
        if not block:
            return labels                                  # 预热期：还没有板块划分
        L = _log_panel(data)
        r = np.zeros((T, n), dtype="float64")
        r[1:] = L[1:] - L[:-1]
        for k, g in enumerate(self._groups_at(L, r, block[-1], corr_win, n)):
            labels[g] = k
        return labels


# ------------------------------------------------------------------ 导出

__all__ = [
    "SsdPairsStrategy",
    "CointPairsPortfolioStrategy",
    "SectorNeutralPairsStrategy",
]
