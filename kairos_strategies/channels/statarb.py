"""渠道：statarb —— 统计套利（市场中性多空，逐行权重之和 ≈ 0）。

收集途径与范式（**全部为本仓库原创实现**，只用 numpy/pandas，不引入
statsmodels / sklearn，协整回归与 PCA 全部手写）：

1. ``coint_pairs``        —— Engle-Granger 两步协整配对：OLS 估对冲比率 β，
   对残差 ``logy - α - β·logx`` 做滚动 z-score，价差被高估则做空价差、被低估则
   做多价差（两腿等预算反向，行和恒为 0）。配对本身在滚动网格上**自动挑选**：
   遍历全部 (i, j) 组合，对残差跑自研 ADF 回归得到 t 统计量与半衰期作为门槛，
   再按「价差最紧(R² 最高) > 协整证据最强(t 最负) > 回归最快(半衰期最短)」排序
   取第一名（详见 ``_resid_stationarity`` 与 ``_select_pair``）。
2. ``eof_stat_arb``      —— PCA / EOF 统计套利（Avellaneda-Wei 思路的自研简化版）：
   对截面收益的滚动协方差做 ``numpy.linalg.eigh`` 特征分解，取前 m 个主成分构成
   投影矩阵 P 重构「公允（系统性）收益路径」，特质残差 ``u = r - P r`` 累计成
   实际价格相对公允价格的偏离，标准化后做多被低估、做空被高估者。
3. ``xs_zscore_reversion`` —— 截面 z-score 短期反转：每期把尾部 lookback 收益做
   截面标准化，做多截面偏弱（z 低）、做空截面偏强（z 高）的一篮子（全篮子多空）。
4. ``basket_neutral``    —— 双篮子相对价值：先用尾部已实现波动把资产分成低波/高波
   两个结构性篮子，再对两篮子的「便宜度」价差做 z-score，做多相对便宜篮子、
   做空相对贵篮子；篮子内部等权、两篮子各占一半预算，天然美元中性。

统一后处理 ``_neutralize``：逐行在**候选资产**内去均值（强制行和 ≈ 0，即美元/市场
中性），再按「每行绝对值之和 ≤ budget(=1)」整体缩放（不加杠杆）。引擎的换手计算
对美元中性组合已用 ``1 + r_p`` 作分母，故行和为 0 不会造成换手虚增。

关于 ADF 检验：本渠道不依赖 statsmodels，自己用 numpy 做 ADF 风格的残差单位根回归
``Δe_t = c + λ·e_{t-1}``（见 ``_resid_stationarity``），并解析求出 ``t = λ / se(λ)``
（一元回归的 ``Var(λ) = σ² / Σ(e_{t-1} - ē)²``）；t 越负表示拒绝「残差有单位根」
（即拒绝不协整）的证据越强，等价于 Engle-Granger 第二步的 τ 统计量。同时给出半衰期
``-ln2 / ln(1 + λ)`` 衡量回归速度。注意：本实现**不查 Dickey-Fuller 临界值表、不给
p 值**，只用 t 做同序排序 + 一个宽松地板阈值（``min_adf_t``），因此这是工程近似而非
严格的统计显著性检验；小样本下伪回归仍可能让残差显得平稳。

防未来函数：协整回归、PCA 载荷、波动率与篮子划分都在「锚定样本起点的再平衡网格」
上用**尾部窗口**估计，同一 block 内系数冻结（只用 ≤ t_rebalance 的数据），t 期权重
只依赖 ≤ t 的信息；因此截断样本重算得到的前缀权重与全样本完全一致（见
``tests/test_statarb.py`` 的 prefix-invariance 测试）。全程离线、无随机数、确定性。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-9          # 价格下限保护，避免 log(0)
_TINY = 1e-12        # 除零 / 退化保护阈值


# ------------------------------------------------------------------ 通用工具

def _log_prices(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64 对数价格（先裁到正数，保证 log 与比值运算安全）。"""
    return np.log(data.prices.astype("float64").clip(lower=_EPS))


def _refit_grid(n_rows: int, start: int, freq: int) -> List[int]:
    """再平衡索引网格：``start, start + freq, ...``（锚定样本起点）。

    锚定在绝对索引 0 而非样本末尾，是「截断样本前缀不变」（防未来函数）的关键：
    加入更多未来数据不会改变历史上任何一次再平衡的时点与内容。
    """
    start = int(start)
    if start < 0 or start >= int(n_rows):
        return []
    return list(range(start, int(n_rows), max(int(freq), 1)))


def _ramp(z: np.ndarray, exit_z: float, entry_z: float) -> np.ndarray:
    """死区 + 线性斜坡的连续仓位强度（滞后带的连续版本，用于压低换手）。

    ``|z| <= exit_z`` -> 0（中枢附近不交易）；``|z| >= entry_z`` -> ±1（满强度）；
    两者之间线性插值，故信号在阈值附近抖动时权重连续变化而非跳变。
    """
    z = np.asarray(z, dtype="float64")
    span = max(float(entry_z) - float(exit_z), _TINY)
    mag = np.clip((np.abs(z) - float(exit_z)) / span, 0.0, 1.0)
    out = np.sign(z) * mag
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _neutralize(raw: pd.DataFrame, data: MarketData, budget: float = 1.0) -> pd.DataFrame:
    """把任意原始信号后处理成「美元中性 + 不加杠杆」的目标权重面板。

    逐行（逐日）在**候选资产**（原始信号有限且非零者）内：
      1. 去截面均值 -> 该行权重之和严格 ≈ 0（市场/美元中性）；
      2. 若绝对值之和 > ``budget`` 则整体等比缩放 -> 每行绝对值和 ≤ 1（无杠杆）；
      3. 候选少于 2 个（凑不出多空两腿）时整行置 0。
    输出对齐 ``data`` 的 index/columns，且不含 NaN。
    """
    aligned = (raw.reindex(index=data.dates, columns=data.symbols)
                  .apply(pd.to_numeric, errors="coerce"))
    v = np.array(aligned.to_numpy(dtype="float64"), copy=True)
    v[~np.isfinite(v)] = 0.0
    mask = v != 0.0
    cnt = mask.sum(axis=1)
    row_mean = np.where(mask, v, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    cen = np.where(mask, v - row_mean[:, None], 0.0)
    abs_sum = np.abs(cen).sum(axis=1)
    scale = np.where(abs_sum > budget,
                     budget / np.where(abs_sum > _TINY, abs_sum, 1.0),
                     1.0)
    out = cen * scale[:, None]
    out[cnt < 2] = 0.0
    out[~np.isfinite(out)] = 0.0
    return pd.DataFrame(out, index=data.dates, columns=data.symbols)


def _nan_col_mean(x: np.ndarray) -> np.ndarray:
    """逐行「忽略 NaN 的均值」：整行无有效值（或列为空）时返回 NaN。"""
    x = np.asarray(x, dtype="float64")
    if x.ndim != 2 or x.shape[1] == 0:
        return np.full(x.shape[0] if x.ndim == 2 else 0, np.nan, dtype="float64")
    fin = np.isfinite(x)
    cnt = fin.sum(axis=1)
    tot = np.where(fin, x, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _ols_ab(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """一元 OLS ``y = alpha + beta * x``，返回 (alpha, beta)（lstsq，确定性）。"""
    n = int(x.size)
    design = np.column_stack([np.ones(n, dtype="float64"), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _resid_stationarity(resid: np.ndarray) -> Tuple[float, float, float]:
    """残差平稳性诊断（ADF / Engle-Granger 第二步的自研 numpy 版，不用 statsmodels）。

    对残差做 ``Δe_t = c + λ·e_{t-1} + ε_t`` 的 OLS（即 ADF 回归，带截距、无滞后差分项），
    返回 ``(rho, half_life, adf_t)``：

    * ``rho = 1 + λ``：残差的 AR(1) 系数，``rho < 1`` 即均值回归；
    * ``half_life = -ln2 / ln(rho)``：回归一半所需期数（``0 < rho < 1`` 时有限，越小越快）；
      ``rho <= 0``（过度回归/振荡）记作 1.0，``rho >= 1``（单位根/爆炸）记作 ``inf``；
    * ``adf_t = λ / se(λ)``：**ADF 风格的 t 统计量**，se 由 OLS 残差方差与
      ``Σ(e_{t-1} - ē)²`` 解析求得（一元回归的 ``Var(λ) = σ² / Σ(e_{t-1}-ē)²``）。
      越负表示拒绝「残差有单位根」（即拒绝不协整）的证据越强，等价于 EG 检验的
      τ 统计量。本实现不查 Dickey-Fuller 临界值表、也不给 p 值，只用它做**同序排序**
      与一个保守阈值过滤（见 ``CointPairsStrategy.params['min_adf_t']``）。

    数据不足或退化时返回 ``(1.0, inf, 0.0)``（视为不协整）。
    """
    e0 = np.asarray(resid, dtype="float64")[:-1]
    de = np.diff(np.asarray(resid, dtype="float64"))
    if e0.size < 16 or not np.isfinite(e0).all() or not np.isfinite(de).all():
        return 1.0, np.inf, 0.0
    sxx = float(((e0 - e0.mean()) ** 2).sum())
    if sxx <= _TINY:
        return 1.0, np.inf, 0.0
    const, lam = _ols_ab(e0, de)
    dof = max(e0.size - 2, 1)
    fit_err = de - (const + lam * e0)
    sigma2 = float(fit_err @ fit_err) / dof
    var_lam = sigma2 / sxx
    adf_t = lam / np.sqrt(var_lam) if var_lam > _TINY else 0.0
    rho = 1.0 + lam
    if not np.isfinite(rho) or rho >= 1.0 - 1e-12:
        return rho, np.inf, float(adf_t)
    if rho <= 0.0:
        return rho, 1.0, float(adf_t)
    return rho, float(-np.log(2.0) / np.log(rho)), float(adf_t)


# ------------------------------------------------------------------ 策略 1：协整配对

class CointPairsStrategy(Strategy):
    name = "coint_pairs"
    channel = "statarb"
    universe = "cross_section"
    long_only = False
    description = ("协整配对统计套利：在滚动网格上遍历全部资产对，用 OLS 做 Engle-Granger 第一步估"
                   "对冲比率 β，第二步对残差跑自研 ADF 回归（Δe=c+λe₋₁）得到 t 统计量与半衰期，"
                   "在通过门槛的对里挑「价差最紧(R² 最高)、协整证据最强」的一对；再对残差 "
                   "logy-α-β·logx 做滚动 z-score，价差被高估(z 高)做空价差、被低估(z 低)做多价差，"
                   "两腿等预算反向，组合逐行权重和恒为 0。")
    hypothesis = ("核心假设：存在协整关系的资产对，其线性组合（相对估值）是平稳的，偏离中枢后由"
                  "套利/基本面纽带拉回，故残差 z-score 的反向持仓期望为正，且多空对冲剥离了市场 "
                  "beta。失效场景：协整关系结构性破裂（并购、行业政策、基本面脱钩）时残差不回归而"
                  "持续发散，亏损无自然上限；ADF/半衰期只刻画历史平稳性，且小样本下伪回归会让残差"
                  "看起来均值回归，故仍可能选中并不真协整的对。")
    source = ("统计套利经典范式——Engle-Granger 协整配对交易（cointegrated pairs trading）。"
              "OLS 对冲比率、ADF 风格 t 统计量、半衰期与配对排序均为本仓库原创 numpy 实现，不依赖 "
              "statsmodels（也不查 Dickey-Fuller 临界值表）；与 meanrev 渠道的 pairs_spread（固定"
              "两腿、β=1 的对数价差）不同，此处配对是自动筛选的且带估出的 β。")
    params = {"est_window": 120,      # 协整回归 / ADF 诊断的尾部窗口
              "z_window": 40,         # 残差滚动 z-score 窗口
              "select_freq": 20,      # 重新选对/重估 β 的间隔（block 内系数冻结）
              "entry_z": 1.5,         # 满强度阈值
              "exit_z": 0.5,          # 死区阈值（|z| 小于它不交易）
              "leg_budget": 0.5,      # 单腿预算：两腿绝对值之和 = |pos| ≤ 1
              "max_half_life": 60.0,  # 半衰期上限：回归太慢的价差没有交易价值
              "min_adf_t": -2.0}      # ADF t 统计量上限（越负越显著；宽松地板，宁缺毋滥）

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero                                    # 单资产无法配对
        L = np.array(_log_prices(data).to_numpy(dtype="float64"), copy=True)
        T = L.shape[0]
        est = int(self.params["est_window"])
        zwin = int(self.params["z_window"])
        start = max(est, zwin) - 1
        if T <= start:
            return zero                                    # 历史不足，整段空仓
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        leg = float(self.params["leg_budget"])
        out = np.zeros((T, n), dtype="float64")
        grid = _refit_grid(T, start, int(self.params["select_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            picked = self._select_pair(L, t0, est, pairs)
            if picked is None:
                continue                                   # 本 block 无合格协整对 -> 空仓
            (xi, yj), alpha, beta = picked
            # 全历史用同一组 (α, β) 计算残差，保证 z-score 的度量单位一致；
            # 滚动 z 只看尾部 zwin 期，故 block 内每个 t 都只用 ≤ t 的信息。
            resid = L[:, yj] - alpha - beta * L[:, xi]
            z = ind.rolling_zscore(pd.Series(resid, index=data.dates),
                                   zwin).to_numpy(dtype="float64")
            # p > 0 = 做多价差（多 y 空 x）；z 高（价差贵）-> p < 0 = 做空价差
            p = -_ramp(z[t0:t1], float(self.params["exit_z"]), float(self.params["entry_z"]))
            out[t0:t1, yj] = leg * p
            out[t0:t1, xi] = -leg * p
        return _neutralize(pd.DataFrame(out, index=data.dates, columns=data.symbols), data, 1.0)

    def _select_pair(self, L: np.ndarray, t0: int, est: int,
                     pairs: Sequence[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], float, float]]:
        """在尾部窗口 [t0-est+1, t0] 上挑出「最像协整、且价差最紧」的资产对。

        入选门槛（同时满足，否则跳过该对）：
          1. 残差非退化（两腿都不是常数、且不完全共线）；
          2. ``adf_t <= min_adf_t``：ADF 风格 t 统计量足够负，即有证据拒绝「残差有单位根」；
          3. ``half_life <= max_half_life``：回归速度够快，价差偏离才有交易价值。
        排序键为 ``(-r2, adf_t, half_life, i, j)`` 升序：先取**水平联动最紧**的一对
        （``r2 = 1 - Σe²/Σ(y-ȳ)²``，即残差方差占 y 方差的比例最小、价差最"干净"），
        r2 并列时再比协整证据强度、回归速度，最后按资产索引字典序，保证并列时结果
        唯一（确定性、可复现）。r2 只参与排序、不设硬门槛，因此在没有真协整关系的
        universe 上仍能挑出相对最紧的一对；是否空仓完全由门槛 2/3 决定。
        全部不合格返回 None -> 该 block 空仓（宁可不交易，也不做不协整的配对）。
        """
        lo = t0 - est + 1
        max_hl = float(self.params["max_half_life"])
        min_t = float(self.params["min_adf_t"])
        best: Optional[Tuple[Tuple[int, int], float, float]] = None
        best_key: Optional[Tuple[float, float, float, int, int]] = None
        for i, j in pairs:
            x = L[lo:t0 + 1, i]
            y = L[lo:t0 + 1, j]
            if x.size < 16 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            syy = float(((y - y.mean()) ** 2).sum())
            if float(x.std()) <= _TINY or syy <= _TINY:
                continue
            alpha, beta = _ols_ab(x, y)
            resid = y - alpha - beta * x
            if float(resid.std()) <= _TINY:
                continue                                   # 完全共线，无残差可交易
            _, hl, adf_t = _resid_stationarity(resid)
            if not np.isfinite(hl) or hl > max_hl:
                continue                                   # 单位根 / 回归太慢
            if not np.isfinite(adf_t) or adf_t > min_t:
                continue                                   # 拒绝单位根的证据不足
            r2 = 1.0 - float(resid @ resid) / syy          # 水平联动紧密度（价差质量）
            key = (-r2, adf_t, hl, int(i), int(j))
            if best_key is None or key < best_key:
                best_key, best = key, ((i, j), alpha, beta)
        return best


# ------------------------------------------------------------------ 策略 2：PCA/EOF 统计套利

class EofStatArbStrategy(Strategy):
    name = "eof_stat_arb"
    channel = "statarb"
    universe = "cross_section"
    long_only = False
    description = ("PCA/EOF 统计套利：对截面收益的滚动协方差做 numpy.linalg.eigh 特征分解，取前 m "
                   "个主成分构造投影矩阵 P 重构「公允（系统性）收益」，特质残差 u = r - P·r 在 h 日"
                   "上累计成实际价格相对公允价格的偏离，按自身残差波动标准化后做多被低估(z 低)、"
                   "做空被高估(z 高)的资产，逐行去均值后为美元中性组合。")
    hypothesis = ("核心假设：截面收益的大部分方差由少数共同因子（主成分/EOF）解释，剔除因子后的"
                  "特质偏离多为流动性冲击与过度反应造成的临时错价，会向因子隐含的公允价格回归，"
                  "且多空对冲掉因子敞口后风险有限。失效场景：偏离来自真实的特质信息（财报、被并购、"
                  "退市风险）时会继续扩大；因子结构不稳定或主成分个数选错（m 过大把信号本身当因子"
                  "吸收、m 过小残留因子敞口）都会显著削弱效果。")
    source = ("Avellaneda & Wei (2010) 的 PCA 统计套利思路（用主成分重构公允价格、对残差做 OU "
              "均值回归）的自研简化版：改为对**收益**（而非对数价格）做滚动 PCA，用「h 日累计残差 / "
              "残差波动」替代 OU 参数估计。特征分解、投影与标准化全部为本仓库原创 numpy 实现，"
              "不使用 sklearn/statsmodels。")
    params = {"window": 150,        # 协方差/主成分估计的尾部窗口
              "n_components": 2,    # 保留的主成分个数 m（自动裁到 [1, N-1]）
              "refit_freq": 5,      # 重新做特征分解的间隔（block 内载荷冻结）
              "horizon": 5,         # 残差累计期 h（错价的观察尺度）
              "z_window": 40,       # 残差波动的尾部估计窗口
              "z_cap": 3.0,         # 标准化偏离的截断，防单一资产吃满预算
              "budget": 1.0}        # 组合毛敞口上限（每行绝对值和）

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero
        T = int(len(data.dates))
        win = int(self.params["window"])
        if T < win + 2:
            return zero
        logp = _log_prices(data)
        r = np.array(logp.diff().to_numpy(dtype="float64"), copy=True)
        r[0, :] = 0.0                                     # 首期无收益，置 0（非未来信息）
        r[~np.isfinite(r)] = 0.0
        m = int(np.clip(int(self.params["n_components"]), 1, max(1, n - 1)))
        h = max(int(self.params["horizon"]), 1)
        u = np.full((T, n), np.nan, dtype="float64")      # 载荷就绪前保持 NaN（不下注）
        grid = _refit_grid(T, win - 1, int(self.params["refit_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            proj = self._projector(r[t0 - win + 1:t0 + 1], m)
            if proj is None:
                continue
            blk = r[t0:t1]
            u[t0:t1] = blk - blk @ proj.T                 # P 对称：(P·r')ᵀ = r'ᵀ·P
        udf = pd.DataFrame(u, index=data.dates, columns=data.symbols)
        drift = udf.rolling(h, min_periods=h).sum()        # h 日累计特质残差 = 错价
        scale = udf.rolling(int(self.params["z_window"]),
                            min_periods=int(self.params["z_window"])).std() * np.sqrt(float(h))
        z = drift / scale.where(scale > _TINY)             # 以自身残差波动为单位的偏离
        z = z.clip(-float(self.params["z_cap"]), float(self.params["z_cap"]))
        return _neutralize(-z, data, float(self.params["budget"]))   # 低估做多、高估做空

    @staticmethod
    def _projector(returns_win: np.ndarray, m: int) -> Optional[np.ndarray]:
        """由尾部收益窗口估计前 m 个主成分的投影矩阵 P = V_m·V_mᵀ。

        先做时序去均值，再对样本协方差用 ``eigh``（对称矩阵，实特征值）分解，按特征值
        **降序稳定排序**取前 m 个特征向量。P 只依赖特征子空间，与特征向量的符号无关，
        因此结果确定、可复现。窗口退化（行数不足 / 含 NaN / 总方差为 0，如价格恒定）
        时返回 None —— 没有可估的因子结构就不下注。
        """
        X = np.asarray(returns_win, dtype="float64")
        if X.ndim != 2 or X.shape[0] < X.shape[1] + 2 or not np.isfinite(X).all():
            return None
        Xc = X - X.mean(axis=0)
        cov = Xc.T @ Xc / float(max(Xc.shape[0] - 1, 1))
        cov = 0.5 * (cov + cov.T)                         # 强制对称，数值稳健
        if not np.isfinite(cov).all() or float(np.trace(cov)) <= _TINY:
            return None                                   # 总方差为 0：窗口无信息
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(-evals, kind="stable")          # 降序 + 稳定排序（确定性）
        if float(evals[order[0]]) <= _TINY:
            return None                                   # 首成分也无方差
        V = evecs[:, order[:max(int(m), 1)]]
        return V @ V.T


# ------------------------------------------------------------------ 策略 3：截面 z-score 反转

class XsZscoreReversionStrategy(Strategy):
    name = "xs_zscore_reversion"
    channel = "statarb"
    universe = "cross_section"
    long_only = False
    description = ("截面 z-score 反转：每期把各资产尾部 lookback 日收益做截面标准化（减截面均值、"
                   "除截面标准差）得到相对强弱 z，做多截面偏弱(z 低)、做空截面偏强(z 高)的一篮子，"
                   "z 经短窗口尾部平滑后按 -z 配权并逐行去均值，构成美元中性的全篮子多空组合。")
    hypothesis = ("核心假设：同一截面内短期相对涨跌包含过度反应与流动性冲击成分，近端相对弱势者"
                  "倾向反弹、相对强势者倾向回吐，故反向持仓可赚取截面收敛；由于同时多空，市场"
                  "beta 被对冲掉，收益来源是截面离散度而非方向。失效场景：截面动量行情（强者恒强的"
                  "行业主题/资金抱团）中反转持续为负；截面资产太少或高度同质时 z 的区分度不足，"
                  "信号被交易成本吞噬。")
    source = ("统计套利经典范式——截面标准化反转（cross-sectional z-score reversal）。本仓库原创"
              "实现：与 factor 渠道的 short_term_reversal（只做多、取排名前 1/3）和 meanrev 渠道的 "
              "zscore_reversion（单资产时序 z、择时）在信号构造与组合形态上均不同，此处是**全篮子"
              "市场中性**的截面版本。")
    params = {"lookback": 10,     # 相对强弱的观察期（尾部收益）
              "smooth": 3,        # z 的尾部平滑期（抑制换手，只用历史）
              "z_cap": 3.0,       # 截面 z 截断，避免单一离群资产吃满预算
              "budget": 1.0}      # 组合毛敞口上限

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        if int(data.n_assets) < 2:
            return pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        prices = data.prices.astype("float64").clip(lower=_EPS)
        lookback = max(int(self.params["lookback"]), 1)
        r = prices / prices.shift(lookback) - 1.0            # 尾部 lookback 日收益（不含未来；
        #                                                     NaN 原样保留，不做隐式前向填充）
        mu = r.mean(axis=1)                                # 截面均值（当期截面信息）
        sd = r.std(axis=1)                                 # 截面标准差（ddof=1）
        z = r.sub(mu, axis=0).div(sd.where(sd > _TINY), axis=0)
        cap = float(self.params["z_cap"])
        z = z.clip(-cap, cap)
        smooth = max(int(self.params["smooth"]), 1)
        if smooth > 1:
            z = z.rolling(smooth, min_periods=smooth).mean()   # 尾部平滑，降低换手
        return _neutralize(-z, data, float(self.params["budget"]))


# ------------------------------------------------------------------ 策略 4：篮子中性

class BasketNeutralStrategy(Strategy):
    name = "basket_neutral"
    channel = "statarb"
    universe = "cross_section"
    long_only = False
    description = ("双篮子相对价值中性：先按尾部已实现波动把资产等分成低波/高波两个结构性篮子（信息"
                   "不足时退回按列索引奇偶划分），再对两篮子「便宜度」（各资产对数价格相对自身中枢的"
                   "滚动 z 的篮子均值）之差做滚动 z-score，价差偏贵则做空该篮子、做多另一篮子；篮子"
                   "内部等权、两篮子各占一半预算，逐行权重和恒为 0。")
    hypothesis = ("核心假设：把资产聚成两个风格篮子后，篮子层面的相对估值（贵/便宜）比单资产噪声"
                  "更稳定，篮子价差围绕中枢波动并会收敛；同时多空两个篮子对冲掉了市场因子，组合对"
                  "大盘方向中性，只承担「篮子间相对价值」这一维风险。失效场景：两个篮子出现结构性"
                  "分化（风格切换、低波与高波长期跑赢关系反转）时价差不收敛；篮子划分本身不稳定"
                  "（波动排名频繁换位）会增加换手与实现偏差。")
    source = ("统计套利经典范式——篮子/板块相对价值多空（basket spread trading、市场中性配对篮子）。"
              "本仓库原创实现：篮子划分用尾部已实现波动排名（结构性因子，与交易信号解耦以避免"
              "自我循环），信号用两篮子便宜度价差的滚动 z-score；与 meanrev 渠道的单对价差、"
              "factor 渠道的只做多排序均不同。")
    params = {"vol_window": 60,      # 篮子划分用的尾部已实现波动窗口
              "z_window": 40,        # 单资产「便宜度」= 对数价格的滚动 z 窗口
              "spread_window": 60,   # 篮子价差自身的滚动 z 窗口
              "basket_freq": 20,     # 重新划分篮子的间隔（block 内篮子冻结）
              "entry_z": 1.0,        # 价差 z 满强度阈值
              "exit_z": 0.25,        # 价差 z 死区阈值
              "leg_budget": 0.5}     # 单篮子预算：两篮子绝对值之和 = |pos| ≤ 1

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = int(data.n_assets)
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if n < 2:
            return zero                                    # 至少两个资产才能分篮
        logp = _log_prices(data)
        zwin = int(self.params["z_window"])
        volwin = int(self.params["vol_window"])
        spread_win = int(self.params["spread_window"])
        cheap = ind.rolling_zscore(logp, zwin).to_numpy(dtype="float64")   # 便宜度（负=相对自身中枢偏低）
        vol = data.rolling_vol(volwin).to_numpy(dtype="float64")
        T = cheap.shape[0]
        start = max(zwin, volwin) - 1
        if T <= start:
            return zero
        out = np.zeros((T, n), dtype="float64")
        leg = float(self.params["leg_budget"])
        grid = _refit_grid(T, start, int(self.params["basket_freq"]))
        for b, t0 in enumerate(grid):
            t1 = grid[b + 1] if b + 1 < len(grid) else T
            basket_a, basket_b = self._split_baskets(vol[t0], n)
            if basket_a.size == 0 or basket_b.size == 0:
                continue
            # 用当前篮子成员在全历史上算价差（度量单位一致），滚动 z 只看尾部窗口
            spread = self._basket_spread(cheap, basket_a, basket_b)
            z = ind.rolling_zscore(pd.Series(spread, index=data.dates),
                                   spread_win).to_numpy(dtype="float64")
            # pos > 0 = 做多篮子 A、做空篮子 B；B 相对更贵(z>0) 时正是这个方向
            pos = _ramp(z[t0:t1], float(self.params["exit_z"]), float(self.params["entry_z"]))
            out[t0:t1, basket_a] = (leg / basket_a.size) * pos[:, None]
            out[t0:t1, basket_b] = (-leg / basket_b.size) * pos[:, None]
        return _neutralize(pd.DataFrame(out, index=data.dates, columns=data.symbols), data, 1.0)

    @staticmethod
    def _basket_spread(cheap: np.ndarray, basket_a: np.ndarray,
                       basket_b: np.ndarray) -> np.ndarray:
        """篮子价差 = B 篮子便宜度均值 - A 篮子便宜度均值（>0 表示 B 相对更贵）。

        NaN（预热期）自动跳过；若某行整篮都无有效值则该行为 NaN（后续当作不下注）。
        """
        return _nan_col_mean(cheap[:, basket_b]) - _nan_col_mean(cheap[:, basket_a])

    @staticmethod
    def _split_baskets(vol_row: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """把资产等分成两个篮子：A = 低波篮子，B = 高波篮子。

        N ≥ 4 且波动信息完整时按尾部已实现波动**升序稳定排名**切半（结构性因子，
        与「便宜度」信号解耦，避免用同一信号既分组又下注的自我循环）；否则退回
        「按列索引奇偶」的确定性划分，保证任意 universe 都能构成两个非空篮子。
        """
        idx = np.arange(n)
        v = np.asarray(vol_row, dtype="float64")
        if n >= 4 and v.shape == (n,) and np.isfinite(v).all() and float(v.std()) > _TINY:
            order = np.argsort(v, kind="stable")
            k = n // 2
            return np.sort(order[:k]), np.sort(order[k:])
        return idx[idx % 2 == 0], idx[idx % 2 == 1]
