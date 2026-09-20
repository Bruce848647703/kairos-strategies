"""渠道：ml_signal —— 机器学习信号（纯 numpy 自研模型 ×4，严格 walk-forward）。

收集途径与范式（**全部为本仓库原创实现**，只用 numpy/pandas，禁止 sklearn /
statsmodels / scipy，四个模型全部手写）：

1. ``ridge_alpha``   —— 岭回归：闭式解 ``β = (XᵀX + αI)⁻¹ Xᵀy``（截距经中心化
   免惩罚），预测下期收益，按预测值截面排名做多靠前的一篮子（long_only）。
2. ``logit_signal``  —— 逻辑回归：IRLS（迭代重加权最小二乘，即牛顿法）+ L2 正则，
   预测「下期上涨」概率，按概率截面排名配置权重（long_only）。
3. ``gbm_alpha``     —— 梯度提升树：决策桩（深度 1）集成，逐轮拟合最小二乘损失的
   负梯度（=残差），分裂阈值用训练集分位数分箱后全量搜索（确定性、无随机），
   预测下期收益，按预测排名做多高分/做空低分（美元中性）。
4. ``knn_alpha``     —— k 近邻：特征按训练集统计标准化后，用欧氏距离找与当期特征
   最接近的 k 个历史训练样本，以其**已实现**的未来收益均值作预测（lexsort 按
   样本序号打破距离并列，保证确定性），按预测做多高分/做空低分（美元中性）。

Walk-forward 时序（关键——无未来函数）：
  * 特征行 ``F_s`` 只含截至 s 收盘已实现的信息：滞后 1/5/21 期收益、21 期已实现
    波动、RSI(14)、价格相对 21 期均线的偏离（全部复用 indicators 自研指标）；
  * 训练样本是 ``(F_s, r_{s+1})`` 对，标签 ``r_{s+1} = p_{s+1}/p_s − 1`` 为「下一期
    收益」；
  * 在第 t_k 个 refit 日：训练集只取 ``s ≤ t_k − 2``（标签最晚到 ``r_{t_k−1}``，
    即**全部在 t_k−1 之前已实现**）；预测用 ``F_{t_k−1}``（t_k−1 收盘的特征），
    预测目标为 t_k 期收益 ``r_{t_k}``；
  * 权重自 t_k 起**冻结持有**到下一个 refit 日（每 ``refit`` 期重训一次；训练窗口
    按策略取滚动或扩展），块内不随价格变化；
  * 因此第 t 行权重只依赖 ``≤ t−1`` 的价格（比「截至 t 期」的契约更保守；引擎还会
    再滞后一期，双重保险）。refit 网格锚定绝对索引 0，且标准化统计量/分箱边界都
    只来自训练窗口 —— 截断样本或篡改未来价格，历史权重逐位不变（见
    tests/test_ml_signal.py 的 tamper / prefix 测试）。

统一权重构造（预测分数 → 目标权重，逐行/逐 refit 块）：
  * long_only（ridge/logit）：截面降序排名（并列按资产序号确定性打破），做多前
    ``top_frac`` 篮子，选中集内按分数线性倾斜（分数越高权重越大，以第 k 名分数为
    基准），逐行归一到和 = 1（每行和 ≤ 1、权重 ≥ 0）；
  * 多空（gbm/knn）：预测分数在有效资产内去截面均值（行和 ≈ 0，美元中性），再
    缩放到绝对值和 = ``budget``(=1，不加杠杆)；有效资产不足 2 个则整行为 0；
  * 预热期（可训练日期数不足 ``min_train_dates``）整行为 0（空仓）。

确定性与性能：全流程无随机数（闭式解 / IRLS / 分箱全量搜索 / lexsort 均确定），
同一 MarketData 多次调用结果逐位一致。n_assets=8、n_days=1000、refit=21 时约
41 次重训，特征/训练矩阵都是小规模的 numpy 运算，单策略回测秒级完成。
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-9          # 权重归一 / 排名的除零保护
_TINY = 1e-12        # 方差 / 绝对值和的退化保护

# 特征全部有效所需的最少历史行数 = max(特征窗口) = 21
# （momentum(21) 自第 21 行起有效；realized_vol(21) 依赖 pct_change，亦自第 21 行起；
#   sma(21) 第 20 行起；rsi(14) 经 fillna 恒有限。）
FEATURE_WARMUP = 21


# ------------------------------------------------------------------ 特征工程

def _feature_stack(data: MarketData) -> np.ndarray:
    """特征面板 ``(d, T, N)``：第 s 行只含截至 s 收盘已实现的信息（不做 shift）。

    六个特征（全部来自价格，复用 indicators）：
      1. 滞后 1 期收益 ``r_s``；
      2. 滞后 5 期动量 ``p_s/p_{s−5} − 1``；
      3. 滞后 21 期动量 ``p_s/p_{s−21} − 1``；
      4. 21 期已实现波动（年化）；
      5. RSI(14)（Wilder 平滑，0~100）；
      6. 价格相对 21 期 SMA 的偏离 ``p_s/sma_s − 1``。

    无未来函数由**调用方**保证：在日期 t 只用第 t−1 行做预测特征、只用 ≤ t−2 的
    行（配 ≤ t−1 的已实现标签）做训练。非有限值统一置 NaN，由样本掩码剔除。
    """
    p = data.prices.astype("float64")
    feats = (
        ind.momentum(p, 1),
        ind.momentum(p, 5),
        ind.momentum(p, 21),
        ind.realized_vol(p, 21, int(data.periods_per_year)),
        ind.rsi(p, 14),
        p / ind.sma(p, 21).replace(0.0, np.nan) - 1.0,
    )
    arr = np.stack([np.asarray(f, dtype="float64") for f in feats])
    arr[~np.isfinite(arr)] = np.nan
    return arr


# ------------------------------------------------------------------ 分数 → 权重

def _long_only_row_weights(scores: np.ndarray, top_frac: float) -> np.ndarray:
    """「越高越好」的一行预测分数 → 只做多权重（和 = 1 或全 0，权重 ≥ 0）。

    截面降序排名（并列按资产序号打破，确定性），做多前 ``k = ceil(top_frac·N)``
    名（不足 k 个有效分数则全选）；以第 k 名分数为基准做线性倾斜
    ``raw = score − bnd + eps``，归一后分数越高权重越大，对离群值稳健。
    NaN 分数（预热 / 特征缺失）一律不选；无任何有效分数则整行为 0。
    """
    s = np.asarray(scores, dtype="float64")
    n = s.size
    valid = np.isfinite(s)
    if n == 0 or not valid.any():
        return np.zeros(n, dtype="float64")
    k = max(1, int(np.ceil(float(top_frac) * n - 1e-9)))
    k = min(k, int(valid.sum()))
    key = np.where(valid, -s, np.inf)                    # 升序排：高分在前，NaN 最后
    order = np.lexsort((np.arange(n), key))              # 并列按资产序号，确定性
    sel = order[:k]
    bnd = float(s[sel].min())                            # 第 k 名分数 = 选中集基准
    raw = np.zeros(n, dtype="float64")
    raw[sel] = s[sel] - bnd + _EPS
    total = float(raw.sum())
    if not np.isfinite(total) or total <= _TINY:
        return np.zeros(n, dtype="float64")
    return np.clip(raw / total, 0.0, 1.0)


def _neutral_row_weights(scores: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """「越高越好」的一行预测分数 → 美元中性多空权重（行和 ≈ 0，Σ|w| = budget）。

    在有效（有限）分数内去截面均值：高分做多、低分做空；再整体缩放到绝对值和
    恰为 ``budget``（不加杠杆）。有效资产 < 2（凑不出多空两腿）或分数完全退化
    （去均值后全 0）时整行为 0。
    """
    s = np.asarray(scores, dtype="float64")
    n = s.size
    valid = np.isfinite(s)
    if n < 2 or int(valid.sum()) < 2:
        return np.zeros(n, dtype="float64")
    cen = np.zeros(n, dtype="float64")
    cen[valid] = s[valid] - float(s[valid].mean())
    abs_sum = float(np.abs(cen).sum())
    if not np.isfinite(abs_sum) or abs_sum <= 1e-14:
        return np.zeros(n, dtype="float64")
    return np.clip(cen * (float(budget) / abs_sum), -float(budget), float(budget))


# ------------------------------------------------------------------ 模型核（纯 numpy 自研）

def _ridge_predict(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray,
                   alpha: float) -> np.ndarray:
    """岭回归闭式解预测（L2 正则最小二乘）。

    截距经训练集中心化免惩罚：``β = (X̃ᵀX̃ + αI)⁻¹ X̃ᵀỹ``（X̃/ỹ 为去训练均值），
    预测 ``ŷ = (x − x̄)β + ȳ``。solve 失败时回退 pinv；仍非有限则退化为均值预测。
    """
    mx = ztr.mean(axis=0)
    my = float(ytr.mean())
    xc = ztr - mx
    gram = xc.T @ xc + float(alpha) * np.eye(ztr.shape[1])
    rhs = xc.T @ (ytr - my)
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:                          # pragma: no cover
        beta = np.linalg.pinv(gram) @ rhs
    if not np.isfinite(beta).all():                        # pragma: no cover
        return np.full(zte.shape[0], my)
    return (zte - mx) @ beta + my


def _logit_predict(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray,
                   l2: float, iters: int) -> np.ndarray:
    """L2 逻辑回归（IRLS/牛顿法）预测「下期上涨」概率。

    标签 ``y = 1{r > 0}``；每轮 ``β ← (XᵀWX + Λ)⁻¹ XᵀWz``，其中 ``W = p(1−p)``
    （下限 1e-6 防溢出）、``z = η + (y − p)/W``；截距不正则（Λ₀₀ ≈ 0），初值取
    训练集上涨频率的 logit。线性预测裁剪到 ±30 保证数值稳定；某轮解非有限则
    保留上一轮系数（确定性）。返回 sigmoid 概率 ∈ (0, 1)。
    """
    n, d = ztr.shape
    yb = (ytr > 0.0).astype("float64")
    x1 = np.column_stack([np.ones(n), ztr])
    pen = np.full(d + 1, float(l2))
    pen[0] = 1e-8                                          # 截距几乎不惩罚
    pi = float(np.clip(yb.mean(), 0.01, 0.99))
    beta = np.zeros(d + 1)
    beta[0] = float(np.log(pi / (1.0 - pi)))
    for _ in range(int(iters)):
        eta = np.clip(x1 @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        wgt = np.maximum(p * (1.0 - p), 1e-6)
        z = eta + (yb - p) / wgt
        hess = x1.T @ (x1 * wgt[:, None]) + np.diag(pen)
        grad = x1.T @ (wgt * z)
        try:
            new_beta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:                      # pragma: no cover
            new_beta = np.linalg.pinv(hess) @ grad
        if not np.isfinite(new_beta).all():                # pragma: no cover
            break
        beta = new_beta
    xte = np.column_stack([np.ones(zte.shape[0]), zte])
    eta_te = np.clip(xte @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-eta_te))


def _gbm_predict(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray,
                 n_estimators: int, learning_rate: float, n_bins: int) -> np.ndarray:
    """最小二乘梯度提升（决策桩集成）预测下期收益。

    基学习器为单特征阈值分裂的决策桩：每个特征的候选阈值取训练集分位数分箱的
    箱边界（``n_bins`` 个箱，重复分位数去重），每轮对全部（特征 × 边界）用
    bincount + cumsum 全量搜索最大化 ``S²左/n左 + S²右/n右``（等价于最小化分裂后
    SSE）的分裂；并列取靠前的特征/边界（确定性）。桩输出为两侧残差均值，预测
    更新 ``+= lr × 桩输出``，残差 ``−= lr × 桩输出``。无可分裂特征时提前停止。
    最终预测 = 训练均值 + lr × Σ 桩输出。无随机数、无子采样，完全确定。
    """
    n, d = ztr.shape
    lr = float(learning_rate)
    edges: List[np.ndarray] = []
    btr: List[np.ndarray] = []
    bte: List[np.ndarray] = []
    grid = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for j in range(d):
        e = np.unique(np.quantile(ztr[:, j], grid)[1:-1])
        edges.append(e)
        btr.append(np.searchsorted(e, ztr[:, j], side="right").astype(np.int64))
        bte.append(np.searchsorted(e, zte[:, j], side="right").astype(np.int64))
    base = float(ytr.mean())
    resid = ytr - base
    stumps: List[Tuple[int, int, float, float]] = []       # (特征, 分裂箱, 左值, 右值)
    for _ in range(int(n_estimators)):
        best_gain, best_j, best_c = -np.inf, -1, -1
        for j in range(d):
            nb = int(edges[j].size) + 1
            sums = np.bincount(btr[j], weights=resid, minlength=nb).astype("float64")
            cnts = np.bincount(btr[j], minlength=nb).astype("float64")
            cs, cn = np.cumsum(sums)[:-1], np.cumsum(cnts)[:-1]
            tot, nn = float(sums.sum()), float(cnts.sum())
            ok = (cn > 0.0) & (nn - cn > 0.0)
            if not ok.any():
                continue
            gain = np.where(ok,
                            cs * cs / np.where(cn > 0.0, cn, 1.0)
                            + (tot - cs) ** 2 / np.where(nn - cn > 0.0, nn - cn, 1.0),
                            -np.inf)
            cj = int(np.argmax(gain))                      # 并列取最左边界
            if gain[cj] > best_gain:                       # 并列取靠前特征
                best_gain, best_j, best_c = float(gain[cj]), j, cj
        if best_j < 0 or not np.isfinite(best_gain):
            break
        left = btr[best_j] <= best_c
        lv = float(resid[left].mean())
        rv = float(resid[~left].mean())
        stumps.append((best_j, best_c, lv, rv))
        resid = resid - lr * np.where(left, lv, rv)
    pred = np.full(zte.shape[0], base)
    for j, c, lv, rv in stumps:
        pred = pred + lr * np.where(bte[j] <= c, lv, rv)
    return pred


def _knn_predict(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray, k: int) -> np.ndarray:
    """k 近邻回归预测：邻居「已实现未来收益」的均值。

    对每个测试点计算到全部训练样本的欧氏距离（特征已按训练集统计标准化），
    ``np.lexsort((样本序号, 距离))`` 排序 —— 距离并列按训练样本序号打破，保证
    确定性；取前 ``k_eff = min(k, n_train)`` 个邻居的标签均值作预测。无参数拟合、
    无随机数。
    """
    n_tr = ztr.shape[0]
    kk = int(min(max(int(k), 1), n_tr))
    diff = zte[:, None, :] - ztr[None, :, :]               # (n_te, n_tr, d)，规模小
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    out = np.empty(zte.shape[0], dtype="float64")
    idx = np.arange(n_tr)
    for i in range(zte.shape[0]):
        order = np.lexsort((idx, d2[i]))
        out[i] = float(ytr[order[:kk]].mean())
    return out


# ------------------------------------------------------------------ walk-forward 基类

class _MlWalkForward(Strategy):
    """ml_signal 渠道公共基类：特征工程 + walk-forward 调度 + 权重构造。

    子类只需给出模型核 ``_fit_predict``（输入为按训练集统计标准化后的训练/测试
    特征与连续标签）与元信息。时序契约见模块 docstring：refit 日 t_k 的训练集
    只用 ``s ≤ t_k−2`` 的样本 ``(F_s, r_{s+1})``（标签全部在 t_k−1 前已实现），
    预测用 ``F_{t_k−1}``，权重冻结持有到下一个 refit 日；第 t 行权重只依赖
    ``≤ t−1`` 的价格，绝无未来函数。
    """

    universe = "cross_section"
    params: Dict = {}

    @abstractmethod
    def _fit_predict(self, ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray) -> np.ndarray:
        """模型核：标准化特征 (n, d) + 标签 (n,) → 测试点预测 (m,)。"""
        raise NotImplementedError

    def first_signal_index(self) -> int:
        """首个可能产生非零权重的行号；之前的行全部为预热空仓。

        ``= FEATURE_WARMUP + min_train_dates + 1``：特征自第 ``FEATURE_WARMUP`` 行
        起有效，refit 日 t_k 需要 ``[FEATURE_WARMUP, t_k−2]`` 内至少
        ``min_train_dates`` 个可训练日期。只依赖参数，与样本长度无关（保证截断
        前缀不变）。
        """
        return FEATURE_WARMUP + max(2, int(self.params["min_train_dates"])) + 1

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices.astype("float64")
        T, N = p.shape
        out = np.zeros((T, N), dtype="float64")
        par = self.params
        refit = max(1, int(par["refit"]))
        min_train = max(2, int(par["min_train_dates"]))
        train_window = par.get("train_window")             # None/0 → 扩展窗口
        long_only = bool(self.long_only)
        t0 = self.first_signal_index()
        if T > t0 and N >= 1:
            feats = _feature_stack(data)                   # (d, T, N)
            rets = p.pct_change().to_numpy(dtype="float64")  # (T, N)，标签面板
            d = int(feats.shape[0])
            min_samples = max(10 * d, 30)
            for t_k in range(t0, T, refit):                # 网格锚定绝对索引 0
                hi = t_k - 2                               # 最后一个可训练特征日
                lo = FEATURE_WARMUP
                if train_window:
                    lo = max(lo, hi - int(train_window) + 1)
                nd = hi - lo + 1
                if nd < min_train:
                    continue
                # 训练集：样本 (F_s, r_{s+1})，s ∈ [lo, hi]；标签最晚 r_{t_k−1}，已实现
                xs = feats[:, lo:hi + 1, :].transpose(1, 2, 0).reshape(nd * N, d)
                ys = rets[lo + 1:hi + 2, :].reshape(nd * N)
                keep = np.isfinite(ys) & np.isfinite(xs).all(axis=1)
                if int(keep.sum()) < min_samples:
                    continue
                xtr, ytr = xs[keep], ys[keep]
                # 预测特征：t_k−1 收盘（严格早于 refit 日）
                xte = feats[:, t_k - 1, :].T               # (N, d)
                pv = np.isfinite(xte).all(axis=1)
                if int(pv.sum()) < (2 if not long_only else 1):
                    continue
                # 标准化：统计量只来自训练窗口（无泄漏）；退化列整列置 0
                mu = xtr.mean(axis=0)
                sd = xtr.std(axis=0)
                degen = sd < 1e-10
                sd = np.where(degen, 1.0, sd)
                ztr = (xtr - mu) / sd
                zte = (xte[pv] - mu) / sd
                if degen.any():
                    ztr[:, degen] = 0.0
                    zte[:, degen] = 0.0
                sc = np.asarray(self._fit_predict(ztr, ytr, zte), dtype="float64")
                row = np.full(N, np.nan)                   # 无效/非有限预测 → NaN → 0 权重
                row[pv] = sc
                if long_only:
                    wrow = _long_only_row_weights(row, float(par.get("top_frac", 1.0 / 3.0)))
                else:
                    wrow = _neutral_row_weights(row, float(par.get("budget", 1.0)))
                out[t_k:min(t_k + refit, T), :] = wrow     # 权重冻结持有到下次 refit
        return pd.DataFrame(out, index=data.dates, columns=data.symbols)


# ------------------------------------------------------------------ 策略 1：岭回归

class RidgeAlphaStrategy(_MlWalkForward):
    """自研岭回归 walk-forward 预测下期收益，截面做多预测靠前的一篮子。"""

    name = "ridge_alpha"
    channel = "ml_signal"
    universe = "cross_section"
    long_only = True
    description = ("自研岭回归（闭式解 (XᵀX+αI)⁻¹Xᵀy）walk-forward 预测下期收益：每 21 期用"
                   "滚动 2 年窗口重训（只用已实现标签），按预测值截面排名做多前约 1/3 篮子，"
                   "选中集内按分数线性加权，权重持有到下次重训。")
    hypothesis = ("滞后收益/动量/波动/RSI/均线偏离等特征对下期收益存在弱而相对稳定的线性可"
                  "预测性；L2 收缩把共线特征与噪声的系数整体压小，以微小偏差换取方差大幅下"
                  "降，使小样本下的截面信号更稳健。当特征与收益的关系转为非线性、市场风格"
                  "急切换或信噪比极低时失效，退化为近似等权噪声权重。")
    source = ("机器学习范式——岭回归（Hoerl–Kennard L2 正则最小二乘，闭式解），"
              "numpy.linalg 全自研实现，不使用任何第三方 ML 库。")
    params = {"alpha": 25.0, "refit": 21, "train_window": 504,
              "min_train_dates": 126, "top_frac": 1.0 / 3.0}

    def _fit_predict(self, ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray) -> np.ndarray:
        return _ridge_predict(ztr, ytr, zte, float(self.params["alpha"]))


# ------------------------------------------------------------------ 策略 2：逻辑回归

class LogitSignalStrategy(_MlWalkForward):
    """自研 L2 逻辑回归（IRLS）walk-forward 预测「下期上涨」概率，按概率截面配置。"""

    name = "logit_signal"
    channel = "ml_signal"
    universe = "cross_section"
    long_only = True
    description = ("自研逻辑回归（IRLS 牛顿迭代 + L2 正则）walk-forward 预测「下期上涨」概率："
                   "每 21 期用滚动 2 年窗口重训（标签为已实现的涨跌方向），按概率截面排名"
                   "做多概率最高的前约 1/3 篮子，概率越高权重越大。")
    hypothesis = ("收益方向（涨/跌）比幅度更易学习：二值标签天然截断极端收益的噪声，sigmoid"
                  "链接把特征组合压缩到 (0,1) 概率，截面概率排序即为择强汰弱的信号。当涨跌"
                  "样本严重失衡、特征分布漂移导致概率失准，或方向本身不可预测（有效市场）"
                  "时失效。")
    source = ("机器学习范式——逻辑回归（极大似然 + IRLS/牛顿法求解，L2 正则），"
              "纯 numpy 自研实现，不使用任何第三方 ML 库。")
    params = {"l2": 5.0, "irls_iters": 12, "refit": 21, "train_window": 504,
              "min_train_dates": 126, "top_frac": 1.0 / 3.0}

    def _fit_predict(self, ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray) -> np.ndarray:
        return _logit_predict(ztr, ytr, zte, float(self.params["l2"]),
                              int(self.params["irls_iters"]))


# ------------------------------------------------------------------ 策略 3：梯度提升

class GbmAlphaStrategy(_MlWalkForward):
    """自研梯度提升决策桩 walk-forward 预测下期收益，按预测做多高分/做空低分。"""

    name = "gbm_alpha"
    channel = "ml_signal"
    universe = "cross_section"
    long_only = False
    description = ("自研梯度提升决策桩集成 walk-forward 预测下期收益：扩展窗口每 21 期重训，"
                   "逐轮拟合残差（分位数分箱 + 全量阈值搜索，确定性），按预测值截面去均值后"
                   "做多高分/做空低分（美元中性，绝对值和 = 1），权重持有到下次重训。")
    hypothesis = ("提升法把大量弱学习器（单特征阈值分裂）叠加，可捕捉线性模型漏掉的单调非线"
                  "性与阈值效应（如 RSI 超买超卖、动量突破）；小学习率 + 有限轮数 + 分箱即天"
                  "然正则。金融序列信噪比低，轮数/箱数过大易过拟合噪声；市场结构突变时历史"
                  "分裂点失效。")
    source = ("机器学习范式——梯度提升树（Friedman GBM，决策桩为基学习器），"
              "分箱、分裂搜索与提升循环纯 numpy 自研，不使用任何第三方 ML 库。")
    params = {"n_estimators": 60, "learning_rate": 0.08, "n_bins": 32,
              "refit": 21, "train_window": None, "min_train_dates": 126, "budget": 1.0}

    def _fit_predict(self, ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray) -> np.ndarray:
        return _gbm_predict(ztr, ytr, zte, int(self.params["n_estimators"]),
                            float(self.params["learning_rate"]), int(self.params["n_bins"]))


# ------------------------------------------------------------------ 策略 4：k 近邻

class KnnAlphaStrategy(_MlWalkForward):
    """自研 kNN walk-forward 预测下期收益，按预测做多高分/做空低分。"""

    name = "knn_alpha"
    channel = "ml_signal"
    universe = "cross_section"
    long_only = False
    description = ("自研 k 近邻回归 walk-forward 预测下期收益：特征按训练集统计标准化后，在"
                   "滚动 1.5 年样本中寻找与当期特征欧氏距离最近的 k=32 个历史点，用其已实现"
                   "未来收益的均值作预测，每 21 期重建邻居库；按预测截面去均值做多高分/"
                   "做空低分（美元中性）。")
    hypothesis = ("「历史会押韵」：动量/波动/RSI 等状态组合相似的历史时点，其后续收益分布"
                  "也相似（非参数模式匹配，不假设函数形式，天然适应非线性）。当特征维数偏"
                  "高致距离失去区分度（维数灾难）、市场非平稳使旧邻居不再代表新环境，或"
                  "k 过小（方差大）/过大（偏差大）时失效。")
    source = ("机器学习范式——k 近邻回归（欧氏距离 + 邻居标签均值，lexsort 确定性打破并列），"
              "纯 numpy 自研实现，不使用任何第三方 ML 库。")
    params = {"k": 32, "refit": 21, "train_window": 378,
              "min_train_dates": 126, "budget": 1.0}

    def _fit_predict(self, ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray) -> np.ndarray:
        return _knn_predict(ztr, ytr, zte, int(self.params["k"]))
