"""渠道：multi_factor —— 复合因子 alpha（多因子合成，cross_section，只做多）。

收集途径与范式（**全部为本仓库原创实现**，只用 numpy/pandas；基础因子、截面
标准化、秩相关 IC、滚动 IC/IR 统计、PCA 特征分解与篮子配权工具全部在本模块内
自实现，不 import 其它渠道的私有函数，不依赖 scipy/sklearn/statsmodels）：

与相邻渠道的区别（避免重复）：
  * ``factor`` 渠道交易**单一**异象（低波动 / 短期反转 / 趋势质量 / 残差动量），
    每个策略只用一个分数排序；
  * ``long_short.quality_ls`` 只是「价格效率 + 低波动」两个成分的**固定主观权重**
    合成，且是美元中性多空；
  * 本渠道把 4 个基础因子先做**截面标准化**再用**数据驱动**的方式合成一个复合
    alpha（等权 / 历史 IC 加权 / 历史 IR 加权 / PCA 第一主成分），只做多复合分
    最高的一篮子（每行权重和 ≈ 1、权重 ≥ 0），研究的是「合成规则」本身。

四个基础因子（全部只用价格与波动构造，不臆造任何基本面数据；均为尾部窗口，
越高越好）：
  1. ``momentum``      动量：过去 ``mom_window`` 期累计收益 ``p_t/p_{t-m}-1``；
  2. ``low_vol``       低波动：``-``尾部窗口年化已实现波动（波动越低分越高）；
  3. ``reversal``      短期反转：``-``过去 ``rev_window`` 期收益（近期跌越多分越高）；
  4. ``trend_quality`` 趋势质量：带符号净位移 / 路径长度（效率比 ∈ [-1,1]，
                       「走得直且向上」分越高）。

四个复合方式（本渠道四个策略）：
  1. ``equalweight_composite``  各因子截面标准化后**等权相加**得复合分；
  2. ``ic_weighted_composite``  按各因子**历史 IC**（walk-forward 秩相关）加权，
                                IC 越高权重越大，允许负权（历史 IC 为负则反向用）；
  3. ``max_ir_composite``       按历史 IC 的**信息比率 IR = mean(IC)/std(IC)** 加权，
                                既要预测力又要稳定性；
  4. ``pca_factor``             对历史窗口内各期「截面标准化因子相关阵」做
                                ``numpy.linalg.eigh`` 特征分解，取**第一主成分载荷**
                                作为复合权重（自动放大多数因子共识的方向、压制
                                与共识相悖的方向）。

防未来函数（关键契约，比「截至当期」更保守）：
  * 因子值在第 s 期只用 ≤ s 的价格；进入复合分前统一再滞后 ``score_lag``(=1) 期，
    故**第 t 行权重只依赖 ≤ t−1 的价格**；
  * IC 的定义是「第 s 期因子值 vs 下一期已实现收益 ``r_{s+1}``」的截面秩相关，
    标签 ``r_{s+1}`` 要到 s+1 才实现，因此 IC 统计量额外滞后 ``label_lag``(=2) 期：
    第 t 行的滚动 IC/IR 只由 ``s ≤ t−2``（标签最晚 ``r_{t−1}``，在 t−1 已实现）
    的样本构成；
  * PCA 载荷由 ``≤ t−1`` 期的截面 z 矩阵滚动估计（每个 z 矩阵只含 ≤ 该期价格），
    与因子值同期，不引入任何未来信息；
  * 所有滚动窗口都是**尾部窗口**、截面标准化/排名都是**逐行（逐日）**的，没有
    任何全样本统计量 —— 篡改第 t 期之后的价格，第 t 期及之前的权重逐位不变。

确定性与配权：无随机数；截面排序并列按列索引稳定打破（``np.lexsort``）；PCA
特征向量按「载荷之和 ≥ 0」定向（和 ≈ 0 时按绝对值最大的载荷为正），消除符号
歧义。复合分越高越好，做多降序前 ``k = ceil(top_frac·n_valid)`` 名，篮子内按
**秩线性倾斜**配权（第 r 名得 ``k-r+1``，归一后每行和 = 1）：分数越高权重越大，
且只用序信息，对复合分的量纲与离群值完全不敏感。预热期（信息不足）整行为 0。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-9        # 名额取整 / 归一容差
_TINY = 1e-12      # 方差 / 范数的退化保护

# 基础因子的固定顺序（复合权重面板的列序，测试与研究记录都以此为准）
FACTOR_ORDER: Tuple[str, ...] = ("momentum", "low_vol", "reversal", "trend_quality")

# 四个策略共享的基础参数（各策略再补自己的合成参数）
_COMMON_PARAMS: Dict[str, object] = {
    "mom_window": 63,            # 动量回看窗口（约一个季度）
    "vol_window": 21,            # 已实现波动窗口（约一个月）
    "rev_window": 5,             # 短期反转窗口（约一周）
    "qual_window": 21,           # 趋势质量（净位移/路径长度）窗口
    "top_frac": 1.0 / 3.0,       # 做多复合分最高的 1/3 篮子
    "xs_norm": "z",              # 截面标准化方式："z" 或 "rank"
    "score_lag": 1,              # 因子值再滞后一期 -> 第 t 行只用 ≤ t-1 价格
}
_IC_PARAMS: Dict[str, object] = {
    "label_lag": 2,              # IC 含标签 r_{s+1}，需额外滞后两期才可用
    "ic_window": 126,            # 滚动 IC 统计的尾部窗口（约半年）
    "ic_min_obs": 42,            # 窗口内最少有效 IC 样本数（不足则空仓）
    "ic_min_assets": 3,          # 计算单期秩相关所需的最少有效资产数
}


# ------------------------------------------------------------------ 基础工具（本模块自实现）

def _prices(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64、裁到正数（保证比值/对数安全），对齐 index/columns。"""
    return data.prices.astype("float64").clip(lower=1e-9)


def _forward_returns(p: pd.DataFrame) -> pd.DataFrame:
    """「下一期已实现收益」面板：第 s 行 = ``p_{s+1}/p_s − 1``（末行 NaN）。

    这是**唯一**用到未来价格的地方，且只作为 IC 的标签：IC_s 必须再滞后
    ``label_lag``(=2) 期（即 s ≤ t−2）才允许进入第 t 行的权重，故第 t 行权重
    只用到 ≤ t−1 已实现的收益，绝无未来函数。
    """
    return p.shift(-1) / p - 1.0


def base_factor_panels(data: MarketData, params: Dict) -> Dict[str, pd.DataFrame]:
    """四个基础因子面板（键 = ``FACTOR_ORDER``，越高越好，全部只用 ≤ 当期价格）。

    全部为尾部窗口构造（``shift`` / ``rolling(min_periods=window)``），预热期
    （信息不足）为 NaN，逐行截面标准化时自动被排除。
    """
    p = _prices(data)
    r = p.pct_change()                                       # 首期 NaN（无昨收），非未来信息
    ppy = float(data.periods_per_year)
    mw = max(int(params["mom_window"]), 1)
    vw = max(int(params["vol_window"]), 2)
    rw = max(int(params["rev_window"]), 1)
    qw = max(int(params["qual_window"]), 2)

    momentum = p / p.shift(mw) - 1.0                         # 过去 mw 期累计收益
    low_vol = -(r.rolling(vw, min_periods=vw).std() * np.sqrt(ppy))   # 负的年化已实现波动
    reversal = -(p / p.shift(rw) - 1.0)                      # 负的近期收益（超跌者分高）
    disp = p - p.shift(qw)                                   # 带符号净位移
    path = p.diff().abs().rolling(qw, min_periods=qw).sum()  # 路径长度（累计绝对变动）
    trend_quality = disp / path.replace(0.0, np.nan)         # 效率比 ∈ [-1, 1]

    raw = {"momentum": momentum, "low_vol": low_vol,
           "reversal": reversal, "trend_quality": trend_quality}
    return {k: raw[k].reindex(index=data.dates, columns=data.symbols)
            .apply(pd.to_numeric, errors="coerce") for k in FACTOR_ORDER}


def cross_sectional_z(x: pd.DataFrame) -> pd.DataFrame:
    """截面标准化：逐行减截面均值、除截面标准差（ddof=1，忽略 NaN）。

    只用**当期截面**信息（逐行独立，与时间无关），故与 ``shift`` 可交换：
    ``z(F.shift(1)) == z(F).shift(1)``。某行有效值不足 2 个或标准差 ≈ 0（整行
    退化）时返回 NaN —— 该行不参与排序（预热期天然空仓）。
    """
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > _TINY), axis=0)


def _xs_normalize(x: pd.DataFrame, mode: str = "z") -> pd.DataFrame:
    """截面标准化：``"z"`` -> z 分数；``"rank"`` -> 百分位排名去中值（[-0.5, 0.5]）。

    两种模式都只用当期截面信息；退化行（有效值 < 2 或截面离散度 ≈ 0）整行 NaN。
    """
    if str(mode).lower() == "rank":
        rk = x.rank(axis=1, method="average", pct=True, na_option="keep") - 0.5
        sd = rk.std(axis=1)
        return rk.where(sd > _TINY)
    return cross_sectional_z(x)


def xs_normalize_stack(panels: Sequence[pd.DataFrame], mode: str = "z") -> np.ndarray:
    """把 F 个因子面板截面标准化后堆成 ``(T, N, F)`` 数组（非有限值统一为 NaN）。"""
    cols = [_xs_normalize(p, mode).to_numpy(dtype="float64") for p in panels]
    stack = np.stack(cols, axis=-1)
    return np.where(np.isfinite(stack), stack, np.nan)


def _shift_stack(z: np.ndarray, lag: int) -> np.ndarray:
    """把 ``(T, N, F)`` 因子数组整体向后滞后 ``lag`` 期（前 lag 行填 NaN）。"""
    lag = max(int(lag), 0)
    out = np.full(np.shape(z), np.nan, dtype="float64")
    T = int(np.shape(z)[0])
    if lag == 0:
        out[:] = z
    elif lag < T:
        out[lag:] = z[:T - lag]
    return out


# ------------------------------------------------------------------ 秩相关 IC（walk-forward）

def _row_ranks(x: pd.DataFrame) -> np.ndarray:
    """逐行（逐日）截面排名，NaN 保持 NaN（``na_option='keep'``）。"""
    return x.rank(axis=1, method="average", na_option="keep").to_numpy(dtype="float64")


def _masked_row_corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> np.ndarray:
    """逐行 masked Pearson 相关系数 ``(T,)``：非有限值成对剔除，样本不足 -> NaN。

    两个输入都是秩矩阵时即 Spearman 秩相关（秩对单调变换不变，故与量纲无关）。
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=1).astype("float64")
    safe_n = np.where(n > 0.0, n, 1.0)
    az = np.where(m, a, 0.0)
    bz = np.where(m, b, 0.0)
    ma = az.sum(axis=1) / safe_n
    mb = bz.sum(axis=1) / safe_n
    da = np.where(m, a - ma[:, None], 0.0)
    db = np.where(m, b - mb[:, None], 0.0)
    num = (da * db).sum(axis=1)
    va = (da * da).sum(axis=1)
    vb = (db * db).sum(axis=1)
    den = np.sqrt(va * vb)
    ok = (n >= float(max(int(min_obs), 2))) & (den > _TINY) & np.isfinite(num)
    out = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    return np.clip(out, -1.0, 1.0)


def ic_series_from_panels(panels: Sequence[pd.DataFrame], fwd: pd.DataFrame,
                          min_assets: int = 3) -> np.ndarray:
    """逐因子单期 IC 面板 ``(T, F)``：``ic[s, j] = spearman(F_j[s], r_{s+1})``。

    只用当期截面秩与下一期已实现收益；标签在第 s+1 期才实现，调用方必须再滞后
    （见 ``lagged_ic_stats``）才能进入权重。有效资产不足或截面退化 -> NaN。
    """
    fr = _row_ranks(fwd)
    T = int(fwd.shape[0])
    out = np.full((T, len(panels)), np.nan, dtype="float64")
    for j, panel in enumerate(panels):
        out[:, j] = _masked_row_corr(_row_ranks(panel), fr, min_assets)
    return out


def factor_ic_series(data: MarketData, params: Dict) -> pd.DataFrame:
    """便捷入口：由 ``MarketData`` 直接算单期 IC 面板（列 = ``FACTOR_ORDER``）。"""
    fac = base_factor_panels(data, params)
    ic = ic_series_from_panels([fac[k] for k in FACTOR_ORDER],
                               _forward_returns(_prices(data)),
                               int(params.get("ic_min_assets", 3)))
    return pd.DataFrame(ic, index=data.dates, columns=list(FACTOR_ORDER))


def lagged_ic_stats(ic: np.ndarray, lag: int, window: int,
                    min_obs: int) -> Tuple[np.ndarray, np.ndarray]:
    """IC 面板 -> 滞后 ``lag`` 期后的尾部滚动 (均值, 标准差)，各 ``(T, F)``。

    先 ``shift(lag)`` 再滚动，保证第 t 行的统计量只由「≤ t−lag 期已实现」的 IC
    构成（本渠道 ``lag = label_lag = 2``，即标签最晚到 ``r_{t−1}``）。窗口内有效
    样本 < ``min_obs`` -> NaN（下游整行空仓）。
    """
    lag = max(int(lag), 0)
    w = max(int(window), int(min_obs), 2)
    mp = max(int(min_obs), 2)
    df = pd.DataFrame(np.where(np.isfinite(ic), ic, np.nan)).shift(lag)
    mean = df.rolling(w, min_periods=mp).mean().to_numpy(dtype="float64")
    std = df.rolling(w, min_periods=mp).std().to_numpy(dtype="float64")
    mean = np.where(np.isfinite(mean), mean, np.nan)
    std = np.where(np.isfinite(std), std, np.nan)
    return mean, std


def ic_weight_panel(ic_mean: np.ndarray, ic_std: np.ndarray, scheme: str = "ic",
                    std_floor: float = 0.02) -> np.ndarray:
    """由滚动 IC 统计量生成 ``(T, F)`` 复合权重（数据驱动，可为负权）。

    ``scheme="ic"``：权重 = 历史 IC 均值（IC 越高权重越大，IC 为负则反向使用）；
    ``scheme="ir"``：权重 = 历史 IC 均值 / 历史 IC 标准差（信息比率，既要预测力
    又要稳定性；标准差取 ``max(std, std_floor)`` 防除零与权重爆炸）。

    逐行按 ``Σ|w| = 1`` 归一（正标量缩放不改变截面排序，仅为数值稳健）；任一因子
    缺历史（NaN）-> 整行 NaN（下游空仓）；``Σ|w| ≈ 0``（IC 全为 0）-> 退回等权，
    避免有历史却无信号的日期莫名空仓。
    """
    m = np.atleast_2d(np.array(ic_mean, dtype="float64", copy=True))
    s = np.atleast_2d(np.array(ic_std, dtype="float64", copy=True))
    m = np.where(np.isfinite(m), m, np.nan)
    s = np.where(np.isfinite(s), s, np.nan)
    if str(scheme).lower() == "ir":
        w = m / np.maximum(s, float(std_floor))
    else:
        w = m
    T, F = int(np.shape(w)[0]), int(np.shape(w)[1])
    if F == 0:
        return np.zeros((T, 0), dtype="float64")
    fin = np.isfinite(w)
    rows_ok = fin.all(axis=1)
    abs_sum = np.where(fin, np.abs(w), 0.0).sum(axis=1)
    scale = np.where(abs_sum > _TINY, abs_sum, 1.0)
    out = np.where(rows_ok[:, None], w / scale[:, None], np.nan)
    degen = rows_ok & (abs_sum <= _TINY)
    if np.any(degen):
        out[degen] = 1.0 / float(F)                           # IC 全为 0 -> 退回等权
    return out


# ------------------------------------------------------------------ PCA 第一主成分载荷

def _orient_sign(v: np.ndarray, sign_rule: str = "sum_nonneg") -> np.ndarray:
    """给特征向量定向，消除 ``±v`` 的固有符号歧义（保证复合 alpha 方向可复现）。

    ``"sum_nonneg"``（默认）：令载荷之和 ≥ 0 —— 使复合方向与「多数因子同向看多」
    的共识一致；和 ≈ 0 时退回按绝对值最大的载荷为正。
    ``"max_loading"``：直接令绝对值最大的载荷为正（并列取最靠前的分量）。
    """
    v = np.asarray(v, dtype="float64")
    if v.size == 0:
        return v
    rule = str(sign_rule).lower()
    if rule == "max_loading":
        return -v if float(v[int(np.argmax(np.abs(v)))]) < 0.0 else v
    ssum = float(v.sum())
    if ssum < 0.0:
        return -v
    if abs(ssum) <= 1e-12 and float(v[int(np.argmax(np.abs(v)))]) < 0.0:
        return -v
    return v


def pca_weight_panel(z_stack: np.ndarray, window: int, min_obs: int,
                     lag: int = 1, sign_rule: str = "sum_nonneg") -> np.ndarray:
    """``(T, N, F)`` 截面标准化因子数组 -> ``(T, F)`` 第一主成分载荷（复合权重）。

    做法（全部只用历史）：
      1. 每个有效日期 s 的 z 矩阵 ``Z_s``（N×F，各列已截面标准化：均值 0、
         标准差 1）给出当期的**因子相关阵** ``A_s = Z_sᵀZ_s / (N−1)``；
      2. 对 ``A_s`` 先滞后 ``lag`` 期再做尾部窗口滚动平均（有效样本 < ``min_obs``
         -> 该期 NaN），得到第 t 期可用的「历史平均因子相关阵」``M_t``；
      3. ``numpy.linalg.eigh``（对称阵，实特征值，升序返回）取最大特征值对应的
         特征向量作为第一主成分载荷 ``v_t`` —— 它指出多数因子「共识」的方向，
         复合 alpha = ``Z_{t-1} · v_t``；
      4. 符号定向（消除特征向量固有的 ±v 歧义，保证确定性）：``sign_rule=
         "sum_nonneg"`` 令 ``Σv ≥ 0``（和 ≈ 0 时退回按最大载荷定向），使 alpha
         方向与「多因子同向看多」的共识一致；``"max_loading"`` 直接令绝对值最大
         的载荷为正。最后单位化（``‖v‖ = 1``）。

    历史不足 / 总方差退化 -> 该行 NaN（下游整行空仓）。
    """
    z = np.asarray(z_stack, dtype="float64")
    if z.ndim != 3:
        raise ValueError("z_stack 必须是 (T, N, F) 三维数组")
    T, N, F = z.shape
    out = np.full((T, F), np.nan, dtype="float64")
    if T == 0 or N < 2 or F < 1:
        return out
    valid = np.isfinite(z).all(axis=(1, 2))                  # 该期截面全部因子有效
    zc = np.where(valid[:, None, None], z, 0.0)
    A = np.einsum("tni,tnj->tij", zc, zc) / float(max(N - 1, 1))
    A[~valid] = np.nan
    w = max(int(window), int(min_obs), 2)
    mp = max(int(min_obs), 2)
    flat = pd.DataFrame(A.reshape(T, F * F)).shift(max(int(lag), 0))
    M = flat.rolling(w, min_periods=mp).mean().to_numpy(dtype="float64").reshape(T, F, F)
    for t in range(T):
        Mt = M[t]
        if not np.isfinite(Mt).all():
            continue                                          # 历史不足 -> 空仓
        Mt = 0.5 * (Mt + Mt.T)                                # 强制对称，数值稳健
        if float(np.trace(Mt)) <= _TINY:
            continue                                          # 总方差为 0：无结构可提
        evals, evecs = np.linalg.eigh(Mt)
        if not (np.isfinite(evals).all() and np.isfinite(evecs).all()):
            continue
        if float(evals[-1]) <= _TINY:
            continue                                          # 首成分也无方差
        v = np.array(evecs[:, -1], dtype="float64")           # eigh 升序 -> 末列 = 最大特征值
        v = _orient_sign(v, sign_rule)                        # 消除 ±v 歧义（确定性）
        nrm = float(np.linalg.norm(v))
        out[t] = v / nrm if nrm > _TINY else np.nan
    return out


# ------------------------------------------------------------------ 复合分 -> 只做多篮子权重

def top_basket_weights(scores: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    """「越高越好」的复合分 -> 只做多、秩线性倾斜的 top 篮子权重面板。

    逐行（逐日，只用当期截面分数）：
      1. 有效（非 NaN）分数降序排序，并列按列索引升序稳定打破（``np.lexsort``，
         结果确定）；
      2. 做多前 ``k = clip(ceil(top_frac · n_valid), 1, n_valid)`` 名；
      3. 篮子内**秩线性倾斜**：第 r 名（r = 1..k）得 ``raw = k − r + 1``，归一后
         每行权重和 = 1、权重 ≥ 0，分数越高权重越大；只用序信息，故对复合分的
         量纲、缩放与离群值完全不敏感（不同合成方案之间天然可比）；
      4. 无有效分数（预热 / 退化）-> 整行 0（空仓）。
    """
    v = np.asarray(scores.to_numpy(dtype="float64"), copy=True)
    v = np.where(np.isfinite(v), v, np.nan)
    T, n = v.shape
    out = np.zeros((T, n), dtype="float64")
    frac = float(top_frac)
    for t in range(T):
        row = v[t]
        valid = np.flatnonzero(np.isfinite(row))
        nv = int(valid.size)
        if nv == 0:
            continue                                          # 预热/退化 -> 空仓
        k = max(1, min(int(np.ceil(nv * frac - _EPS)), nv))
        order = np.lexsort((valid, -row[valid]))              # 主键分数降序，次键列索引
        sel = valid[order[:k]]
        raw = np.arange(k, 0, -1, dtype="float64")            # k, k-1, ..., 1
        out[t, sel] = raw / raw.sum()
    return pd.DataFrame(out, index=scores.index, columns=scores.columns)


# ------------------------------------------------------------------ 公共基类

class _CompositeStrategy(Strategy):
    """multi_factor 渠道公共基类：基础因子 -> 截面标准化 -> 复合权重 -> top 篮子。

    子类只需给出复合权重方案 ``_weight_array``（等权 / IC / IR / PCA）与元信息。
    时序契约见模块 docstring：第 t 行权重只依赖 ≤ t−1 的价格与 ≤ t−1 已实现的
    收益（引擎还会再滞后一期，双重保险）。中间量（因子面板、IC 统计、复合权重、
    复合分）都以公开方法暴露，便于研究记录与测试逐项核对。
    """

    universe = "cross_section"
    long_only = True
    weighting = "equal"
    params: Dict = dict(_COMMON_PARAMS)

    # ---- 中间量（可被测试/研究直接调用）----

    def factor_panels(self, data: MarketData) -> Dict[str, pd.DataFrame]:
        """四个基础因子的原始面板（键 = ``FACTOR_ORDER``，越高越好）。"""
        return base_factor_panels(data, self.params)

    def factor_z_stack(self, data: MarketData) -> np.ndarray:
        """截面标准化后的因子数组 ``(T, N, F)``（未滞后，仅含 ≤ 当期价格）。"""
        fac = self.factor_panels(data)
        return xs_normalize_stack([fac[k] for k in FACTOR_ORDER],
                                  str(self.params.get("xs_norm", "z")))

    def _weight_array(self, data: MarketData, fac: Dict[str, pd.DataFrame],
                      z: np.ndarray) -> np.ndarray:
        """复合权重面板 ``(T, F)``（子类实现；严格只用 ≤ t−1 已实现信息）。"""
        raise NotImplementedError

    def factor_weights(self, data: MarketData) -> pd.DataFrame:
        """第 t 期实际使用的复合权重（列 = ``FACTOR_ORDER``），供核对与研究记录。"""
        fac = self.factor_panels(data)
        z = xs_normalize_stack([fac[k] for k in FACTOR_ORDER],
                               str(self.params.get("xs_norm", "z")))
        w = self._weight_array(data, fac, z)
        return pd.DataFrame(np.asarray(w, dtype="float64"),
                            index=data.dates, columns=list(FACTOR_ORDER))

    def composite_scores(self, data: MarketData) -> pd.DataFrame:
        """复合 alpha 分数面板（越高越好）= ``Σ_j w_{j,t} · z_{j,t-1}``。"""
        fac = self.factor_panels(data)
        z = xs_normalize_stack([fac[k] for k in FACTOR_ORDER],
                               str(self.params.get("xs_norm", "z")))
        w = np.asarray(self._weight_array(data, fac, z), dtype="float64")
        zs = _shift_stack(z, int(self.params.get("score_lag", 1)))
        score = np.einsum("tij,tj->ti", zs, w)               # NaN 自然传播（预热 -> 空仓）
        score = np.where(np.isfinite(score), score, np.nan)
        return pd.DataFrame(score, index=data.dates, columns=data.symbols)

    def first_active_index(self) -> int:
        """首个可能非零权重的行号（之前整行空仓）；只依赖参数，与样本长度无关。

        基础因子的最长预热 = ``max(各因子窗口)``（动量 63 期最慢）；再叠加
        ``score_lag``（因子滞后一期）与合成方案自身的历史要求：IC/IR 需要
        ``label_lag + ic_min_obs − 1``，PCA 需要 ``pca_min_obs − 1``。
        """
        p = self.params
        base = max(int(p["mom_window"]), int(p["vol_window"]),
                   int(p["rev_window"]), int(p["qual_window"]))
        lag = int(p.get("score_lag", 1))
        if self.weighting == "pca":
            return base + lag + int(p["pca_min_obs"]) - 1
        if self.weighting in ("ic", "ir"):
            return base + int(p["label_lag"]) + int(p["ic_min_obs"]) - 1
        return base + lag

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        return top_basket_weights(self.composite_scores(data),
                                  float(self.params["top_frac"]))


class _IcLikeComposite(_CompositeStrategy):
    """IC / IR 加权的公共实现：walk-forward 秩相关 IC -> 滚动统计 -> 复合权重。"""

    params: Dict = {**_COMMON_PARAMS, **_IC_PARAMS, "ic_scheme": "ic",
                    "ir_std_floor": 0.02}

    def ic_stats(self, data: MarketData) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """第 t 期可用的（滞后滚动）IC 均值与 IC 标准差面板，列 = ``FACTOR_ORDER``。"""
        fac = self.factor_panels(data)
        ic = ic_series_from_panels([fac[k] for k in FACTOR_ORDER],
                                   _forward_returns(_prices(data)),
                                   int(self.params["ic_min_assets"]))
        mean, std = lagged_ic_stats(ic, int(self.params["label_lag"]),
                                    int(self.params["ic_window"]),
                                    int(self.params["ic_min_obs"]))
        cols = list(FACTOR_ORDER)
        return (pd.DataFrame(mean, index=data.dates, columns=cols),
                pd.DataFrame(std, index=data.dates, columns=cols))

    def _weight_array(self, data: MarketData, fac: Dict[str, pd.DataFrame],
                      z: np.ndarray) -> np.ndarray:
        ic = ic_series_from_panels([fac[k] for k in FACTOR_ORDER],
                                   _forward_returns(_prices(data)),
                                   int(self.params["ic_min_assets"]))
        mean, std = lagged_ic_stats(ic, int(self.params["label_lag"]),
                                    int(self.params["ic_window"]),
                                    int(self.params["ic_min_obs"]))
        return ic_weight_panel(mean, std, str(self.params["ic_scheme"]),
                               float(self.params["ir_std_floor"]))


# ------------------------------------------------------------------ 策略 1：等权复合

class EqualWeightCompositeStrategy(_CompositeStrategy):
    """四个基础因子截面标准化后等权相加，做多复合分最高的一篮子。"""

    name = "equalweight_composite"
    channel = "multi_factor"
    universe = "cross_section"
    long_only = True
    weighting = "equal"
    description = ("等权复合因子：动量、低波动、短期反转、趋势质量四个纯价格因子先做截面标准化"
                   "（z 分数，可选百分位排名），再等权相加成一个复合 alpha，做多复合分最高的"
                   "约 1/3 篮子，篮子内按秩线性倾斜配权（分数越高权重越大），每行权重和 ≈ 1。")
    hypothesis = ("成因：单一异象各自只捕捉收益截面的一维（信息扩散、彩票偏好、流动性冲击、"
                  "趋势干净度），噪声大、时灵时不灵；把方向一致的多个弱信号等权相加，各因子的"
                  "特质噪声在求和中相互抵消，复合分的信噪比与截面区分度高于任何单一成分，且"
                  "等权不需要估计任何参数，避免了权重估计误差（DeMiguel 的 1/N 稳健性结论在"
                  "因子层同样成立）。失效条件：几个因子高度相关时等权等于给同一维暴露加倍下注，"
                  "分散化是假的；因子风格集体逆风（动量崩溃 + 高波反弹同时发生）时四路信号同向"
                  "亏损；等权对量纲/分布异常敏感（已用截面标准化缓解），且在因子有效性差异悬殊"
                  "时明显劣于按 IC/IR 加权。")
    source = ("多因子合成最基础的范式——因子等权（equal-weight factor blending），本仓库原创"
              "实现；四个基础因子与截面标准化、篮子配权工具均在本模块内自实现。与 factor 渠道"
              "的单因子策略（low_volatility / short_term_reversal / trend_quality 各自只用一个"
              "分数）、long_short.quality_ls（两成分固定主观权重的多空版）不同，此处是四因子"
              "标准化后的等权复合、只做多。")
    params = {**_COMMON_PARAMS, "n_factors": len(FACTOR_ORDER),
              "factor_weight": 1.0 / len(FACTOR_ORDER)}

    def _weight_array(self, data: MarketData, fac: Dict[str, pd.DataFrame],
                      z: np.ndarray) -> np.ndarray:
        T = int(len(data.dates))
        F = len(FACTOR_ORDER)
        return np.full((T, F), float(self.params["factor_weight"]), dtype="float64")


# ------------------------------------------------------------------ 策略 2：IC 加权复合

class IcWeightedCompositeStrategy(_IcLikeComposite):
    """按各因子 walk-forward 历史 IC 加权合成（IC 越高权重越大，可为负权）。"""

    name = "ic_weighted_composite"
    channel = "multi_factor"
    universe = "cross_section"
    long_only = True
    weighting = "ic"
    description = ("IC 加权复合：对每个基础因子逐日计算它与下一期已实现收益的截面秩相关（Spearman "
                   "IC），取截至 t−2 的滚动窗口（默认半年）IC 均值作为该因子在第 t 期的复合权重"
                   "（IC 越高权重越大，IC 为负则给负权反向使用），加权合成复合 alpha 后做多最高"
                   "的约 1/3 篮子。")
    hypothesis = ("成因：因子有效性随市场风格漂移，历史 IC 是「这个因子最近还能不能预测收益」的"
                  "直接样本内证据，按 IC 配权等于把预算动态挪到当前有效的因子上，并能自动反向"
                  "使用已经翻转的因子（负 IC -> 负权），比固定等权更自适应；秩相关对离群值与量纲"
                  "不敏感，估计相对稳健。失效条件：IC 本身是噪声很大的统计量（半年窗口约 126 个"
                  "重叠样本、且截面只有几只资产），IC 均值可能只是运气，加权反而放大近期表现最好"
                  "的因子——在风格急切换处会追高杀低（IC 动量崩溃）；因子间相关性高时 IC 加权会"
                  "重复计入同一维暴露；样本太短或截面太窄时 IC 估计偏差大，退化为噪声权重。")
    source = ("多因子合成经典范式——IC 加权（information-coefficient weighting，Grinold-Kahn "
              "的「基本面定律」思路：IR ≈ IC·√breadth，故按 IC 配置因子预算），本仓库原创 "
              "numpy/pandas 实现（逐日截面秩相关 + 滞后滚动统计，walk-forward，不用 scipy/"
              "sklearn）。与 equalweight_composite 的固定等权不同，权重完全由历史 IC 数据驱动。")
    params = {**_COMMON_PARAMS, **_IC_PARAMS, "ic_scheme": "ic", "ir_std_floor": 0.02}


# ------------------------------------------------------------------ 策略 3：最大 IR 复合

class MaxIrCompositeStrategy(_IcLikeComposite):
    """按各因子历史 IC 的信息比率 IR = mean(IC)/std(IC) 加权合成。"""

    name = "max_ir_composite"
    channel = "multi_factor"
    universe = "cross_section"
    long_only = True
    weighting = "ir"
    description = ("最大 IR 复合：对每个基础因子用截至 t−2 的滚动窗口同时估计 IC 均值与 IC 标准差，"
                   "以信息比率 IR = mean(IC)/std(IC)（标准差设下限防爆炸）作为该因子在第 t 期的"
                   "复合权重，IR 越高（预测力既强又稳）权重越大，可为负权；加权合成后做多复合分"
                   "最高的约 1/3 篮子。")
    hypothesis = ("成因：只看 IC 均值会选中「偶尔爆发、大部分时间无效」的因子——它的 IC 均值可能"
                  "不低，但方差极大，实盘贡献不稳定；用 IC 均值除以 IC 标准差（信息比率）配权，"
                  "等价于在因子层面做夏普最大化，惩罚时灵时不灵的因子、奖励持续有效的因子，复合"
                  "alpha 的时序稳定性与风险调整后表现优于纯 IC 加权，本质是给因子权重加了一层"
                  "「方差惩罚」。失效条件：IC 标准差在窗口内被少数极端值主导时 IR 会剧烈抖动；"
                  "因子 IC 接近常数（std ≈ 0）时 IR 趋于无穷，需靠标准差下限兜底；当高 IR 因子"
                  "恰好是低波动、低收益的防御因子时，组合会在牛市里系统性跑输；风格切换处 IR "
                  "与 IC 一样滞后。")
    source = ("多因子合成范式——最大化信息比率（max-IR / Sharpe 型因子配权，Grinold-Kahn 主动"
              "管理框架里的因子层版本），本仓库原创 numpy/pandas 实现：滚动 IC 均值与标准差均"
              "walk-forward 计算（标签滞后两期），不依赖任何优化器。与 ic_weighted_composite "
              "的差别只在权重的分母——多除一个 IC 标准差，用来惩罚不稳定的因子。")
    params = {**_COMMON_PARAMS, **_IC_PARAMS, "ic_scheme": "ir", "ir_std_floor": 0.02}


# ------------------------------------------------------------------ 策略 4：PCA 第一主成分

class PcaFactorStrategy(_CompositeStrategy):
    """对历史窗口内的因子相关阵做 PCA，取第一主成分载荷作为复合权重。"""

    name = "pca_factor"
    channel = "multi_factor"
    universe = "cross_section"
    long_only = True
    weighting = "pca"
    description = ("PCA 复合因子：把四个基础因子的截面标准化值逐日堆成 N×F 矩阵，其 ZᵀZ/(N−1) 即"
                   "当期因子相关阵；对截至 t−1 的滚动窗口（默认半年）平均相关阵做 numpy.linalg."
                   "eigh 特征分解，取最大特征值对应的特征向量（按 Σv ≥ 0 定向、单位化）作为复合"
                   "权重，复合 alpha = 当期截面 z 向量在该方向上的投影，做多最高的约 1/3 篮子。")
    hypothesis = ("成因：多个价格因子之间往往被同一个潜在的市场/风格维度驱动（例如趋势行情里动量"
                  "与趋势质量同涨同跌），第一主成分正是这个共同维度的方向，用它做复合权重相当于"
                  "让数据自己决定「哪些因子在讲同一件事、该合并放大」，无需 IC 标签、无需主观"
                  "先验，且对因子间相关结构的变化自适应；相比 IC/IR 加权，它不依赖收益标签，"
                  "估计噪声更小、换手更低。失效条件：主成分是「方差最大」方向而非「收益最高」"
                  "方向——当共同维度恰好是与未来收益无关的噪声（或所有因子等方差且互不相关时"
                  "相关阵退化为单位阵，主成分方向任意）时，复合 alpha 会退化成某个 arbitrary "
                  "的因子组合；因子相关结构突变（危机中相关性趋同）会让载荷急转、产生跳变换手；"
                  "特征向量符号本身有歧义，必须靠定向规则固定（此处按 Σv ≥ 0）。")
    source = ("多因子合成范式——因子层 PCA / 主成分 alpha（statistical risk model 的主成分思想"
              "反过来用作 alpha 合成方向），本仓库原创 numpy 实现：逐日因子相关阵 + 尾部滚动"
              "平均 + ``numpy.linalg.eigh`` 特征分解 + 确定性符号定向，不用 sklearn/scipy。与 "
              "statarb.eof_stat_arb（对**收益**做 PCA 提取因子、再对残差做均值回归的多空策略）"
              "完全不同：此处 PCA 作用在**因子截面标准化值**上，输出的是复合权重而非公允价格。")
    params = {**_COMMON_PARAMS, "pca_window": 126, "pca_min_obs": 42,
              "n_components": 1, "sign_rule": "sum_nonneg"}

    def _weight_array(self, data: MarketData, fac: Dict[str, pd.DataFrame],
                      z: np.ndarray) -> np.ndarray:
        if int(self.params["n_components"]) != 1:
            raise ValueError("pca_factor 只取第一主成分（n_components 必须为 1）")
        return pca_weight_panel(z, int(self.params["pca_window"]),
                                int(self.params["pca_min_obs"]),
                                int(self.params.get("score_lag", 1)),
                                str(self.params["sign_rule"]))


STRATEGIES: List[type] = [EqualWeightCompositeStrategy, IcWeightedCompositeStrategy,
                          MaxIrCompositeStrategy, PcaFactorStrategy]
