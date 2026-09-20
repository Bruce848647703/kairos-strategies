"""渠道：allocation —— 组合配置/权重构造策略（cross_section，只做多，满仓）。

收集途径：组合管理经典配置范式——等权日历再平衡、逆波动率加权、风险平价
(ERC)、全局最小方差 (GMV)、最大分散化比率 (MDP)。全部为本仓库原创实现：
协方差/波动用 ``data.returns()`` 的滚动样本窗口估计；风险平价用自研牛顿
迭代（对数障碍凸形式），最大分散化用自研两阶段迭代（阻尼定点迭代 +
Frank-Wolfe 精确线搜索），均为纯 numpy，不依赖 scipy/sklearn；最小方差
用 ``np.linalg.solve``（奇异时回退 pinv）+ 负值裁剪投影。

统一范式（防未来函数）：
  第 t 期权重只用截至 t 的收益率窗口 ``[t-window+1, t]`` 估计协方差；
  窗口不足（t < window-1）回退等权 1/N。引擎还会再滞后一期，双重保险。
  所有策略 long_only（权重 ≥ 0）且逐行归一：每行权重和 ≈ 1（满仓配置）。
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_TINY = 1e-18


# ----------------------------------------------------------------------
# 通用小工具：收益率矩阵 / 滚动协方差 / 正则化
# ----------------------------------------------------------------------

def _returns_values(data: MarketData) -> np.ndarray:
    """收益率面板 → 有限 numpy 矩阵（NaN 置 0；价格恒正故收益 > -1）。"""
    vals = data.returns().to_numpy(dtype=float)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_covariances(vals: np.ndarray, window: int) -> np.ndarray:
    """逐日滚动样本协方差张量，形状 (n_days, N, N)。

    第 t 块仅用收益率行 ``[t-window+1, t]``（截至当期、含 t，不看未来），
    ddof=1 无偏样本协方差；t < window-1 时为 NaN，由上层回退等权。
    """
    n, N = vals.shape
    out = np.full((n, N, N), np.nan)
    for t in range(window - 1, n):
        block = vals[t - window + 1: t + 1]
        out[t] = np.atleast_2d(np.cov(block, rowvar=False, ddof=1))
    return out


def _regularize(S: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """对称化 + 岭正则：Σ_reg = Σ + ridge·mean(diag Σ)·I，保证数值正定。

    样本协方差在资产近似共线时会病态/奇异，微小岭项既稳定 solve/迭代，
    又几乎不改变估计本身（相对量级 ~1e-10）。
    """
    S = 0.5 * (S + S.T)
    N = S.shape[0]
    scale = float(np.trace(S)) / max(N, 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return S + max(ridge * scale, _TINY) * np.eye(N)


# ----------------------------------------------------------------------
# 自研求解器：输入正定协方差 Σ，输出 long-only、和=1 的权重向量
# ----------------------------------------------------------------------

def _inverse_vol_weights(S: np.ndarray) -> np.ndarray:
    """逆波动率权重：w ∝ 1/σ，σ 取正则化协方差对角线开方，归一到和=1。"""
    sigma = np.sqrt(np.clip(np.diag(S), _TINY, None))
    w = 1.0 / sigma
    return w / float(w.sum())


def _risk_parity_weights(S: np.ndarray, max_iters: int = 50,
                         tol: float = 1e-10) -> np.ndarray:
    """等风险贡献 (ERC/风险平价) 权重：自研牛顿迭代（对数障碍凸形式）。

    目标：各资产成分风险贡献 ``RC_j = w_j·(Σw)_j`` 相等。等价于求解凸问题
    ``min_w f(w) = ½·wᵀΣw − c·Σ_j ln w_j``（w > 0）：一阶条件 Σw = c/w
    即 ``RC_j = c`` 全相等。用带回溯线搜索的牛顿法求解（Hessian = Σ +
    c·diag(1/w²) 恒正定），对数障碍天然保证 w > 0（long-only），对任意
    正定 Σ（含负相关）全局收敛；c 取 mean(diag Σ) 改善数值尺度，最后归一
    到和=1（ERC 解对 c 只差一个正的缩放，归一后不变）。
    """
    N = S.shape[0]
    if N == 1:
        return np.ones(1)
    c = max(float(np.mean(np.diag(S))), _TINY)

    def _f(w: np.ndarray) -> float:
        return 0.5 * float(w @ (S @ w)) - c * float(np.sum(np.log(w)))

    w = np.ones(N)
    f_cur = _f(w)
    for _ in range(max_iters):
        g = S @ w - c / w                       # 梯度
        H = S + np.diag(c / np.square(w))       # Hessian（恒正定）
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ (-g)
        dec = float(g @ step)                   # = -gᵀH⁻¹g ≤ 0（牛顿减量²的负值）
        if not np.isfinite(dec) or -dec <= tol * max(1.0, abs(f_cur)):
            break                               # 已收敛（或数值退化，保底返回）
        alpha, accepted = 1.0, False
        for _ls in range(60):                   # 回溯线搜索（Armijo + 保持 w>0）
            w_try = w + alpha * step
            if np.all(w_try > 0.0):
                f_try = _f(w_try)
                if np.isfinite(f_try) and f_try <= f_cur + 1e-4 * alpha * dec:
                    w, f_cur, accepted = w_try, f_try, True
                    break
            alpha *= 0.5
        if not accepted:
            break
    total = float(w.sum())
    if not np.isfinite(w).all() or w.min() < 0.0 or total <= _TINY:
        return np.full(N, 1.0 / N)
    return w / total


def _min_variance_weights(S: np.ndarray) -> np.ndarray:
    """全局最小方差权重（long-only 投影）。

    无约束解 ``w ∝ Σ⁻¹1``：用 ``np.linalg.solve``（奇异时回退 ``pinv``）。
    若解含负分量，则裁剪到 0 后重新归一——简单投影，可能牺牲少量方差
    以换取 long-only 满仓；若全被裁剪（病态输入）回退等权。
    """
    N = S.shape[0]
    ones = np.ones(N)
    try:
        x = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(S) @ ones
    if not np.isfinite(x).all():
        return np.full(N, 1.0 / N)
    w = np.clip(x, 0.0, None)
    total = float(w.sum())
    if total <= _TINY:
        return np.full(N, 1.0 / N)
    return w / total


def _max_div_weights(S: np.ndarray, max_iters: int = 80,
                     gamma: float = 0.5, tol: float = 1e-7,
                     fw_iters: int = 60, fw_tol: float = 1e-9,
                     clip_c: float = 10.0) -> np.ndarray:
    """最大分散化 (MDP) 权重：自研两阶段迭代法。

    目标：最大化分散化比率 ``DR = (Σ_j w_j σ_j) / sqrt(wᵀΣw)``。DR 对 w
    零次齐次（尺度不变），故在仿射面 ``{σᵀw = 1, w ≥ 0}`` 上最大化 DR
    等价于凸问题 ``min ½·wᵀΣw s.t. σᵀw = 1, w ≥ 0``（DR = 1/sqrt(2·obj)）。

    阶段一（阻尼定点迭代）：内点最优的一阶条件为 ``(Σw)_j ∝ σ_j``，即
    ``r_j = σ_j/(Σw)_j`` 全相等。迭代 ``w ← w ⊙ clip(r/geo(r))^gamma`` 后
    归一——除以几何均值做尺度中心化（归一化吸收统一缩放，不动点不变），
    裁剪限步长并防止负相关导致 (Σw)_j ≤ 0 时出现 NaN（此时该资产降低
    组合风险，按大比率加仓是正确的上升方向）。对非负相关的良性结构收敛
    快且精度高；收敛判据用未裁剪 r/geo(r) 的相对离散度 < tol（防伪收敛）。

    阶段二（Frank-Wolfe 条件梯度，热启动）：把阶段一结果投影到 σᵀw=1
    面上，迭代「线性子问题取顶点 e_j/σ_j（j = argmin (Σw)_j/σ_j）+ 精确
    线搜索步长」，凸组合更新天然保持 w ≥ 0 且 σᵀw = 1；对偶间隙小于
    fw_tol·obj 时停止。对含负相关、最优解落在边界（部分权重为 0）的情形
    仍能收敛到全局最优，保证 DR 不低于阶段一结果。

    两阶段均为确定性迭代，乘法/凸组合更新下权重恒正（long-only 天然满足）。
    """
    N = S.shape[0]
    if N == 1:
        return np.ones(1)
    sigma = np.sqrt(np.clip(np.diag(S), _TINY, None))

    # ---- 阶段一：阻尼定点迭代 ----
    w = np.full(N, 1.0 / N)
    converged = False
    for _ in range(max_iters):
        y = S @ w
        if not np.isfinite(y).all():
            break
        pos = y > 0.0
        if not pos.any():
            break
        base = sigma / np.where(pos, y, 1.0)
        if pos.all():
            r_raw = base
        else:
            big = 1e6 * max(float(base[pos].max()), 1.0)
            r_raw = np.where(pos, base, big)
        geo = float(np.exp(np.mean(np.log(r_raw))))
        if not np.isfinite(geo) or geo <= 0.0:
            break
        rel = r_raw / geo
        w_new = w * np.power(np.clip(rel, 1.0 / clip_c, clip_c), gamma)
        total = float(w_new.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        w = w_new / total
        if float(np.max(np.abs(rel - 1.0))) < tol:
            converged = True
            break

    # ---- 阶段二：Frank-Wolfe（凸等价问题，热启动）----
    # 阶段一收敛 ⇒ (Σw)_j ∝ σ_j 对所有 j 成立 ⇒ 满足凸 QP 的 KKT 条件
    # （不等式约束 w ≥ 0 不激活），已是全局最优，无需再跑 FW。
    scale = float(sigma @ w)
    if not converged and np.isfinite(scale) and scale > _TINY:
        wf = np.clip(w / scale, 0.0, None)      # 投影到 σᵀw=1 面（DR 不变）
        obj = 0.5 * float(wf @ (S @ wf))
        best_w, best_obj = wf.copy(), obj
        if np.isfinite(obj):
            for _ in range(fw_iters):
                g = S @ wf
                if not np.isfinite(g).all():
                    break
                j = int(np.argmin(g / sigma))   # 线性子问题：顶点 e_j/σ_j
                v = np.zeros(N)
                v[j] = 1.0 / sigma[j]
                d = v - wf
                gap = float(g @ (wf - v))       # FW 对偶间隙 ≥ 0
                if gap <= fw_tol * max(abs(best_obj), _TINY):
                    break
                Sd = S @ d
                a = float(d @ Sd)
                b = float(wf @ Sd)
                eta = float(np.clip(-b / a, 0.0, 1.0)) if a > _TINY else 0.0
                wf = np.clip(wf + eta * d, 0.0, None)
                obj = 0.5 * float(wf @ (S @ wf))
                if np.isfinite(obj) and obj < best_obj:
                    best_w, best_obj = wf.copy(), obj
        # 取两阶段中 DR 更高者（DR 尺度不变，直接在原向量上比）
        def _dr(x: np.ndarray) -> float:
            q = float(x @ (S @ x))
            return float(sigma @ x) / np.sqrt(q) if q > _TINY else 0.0
        if _dr(best_w) > _dr(w):
            w = best_w

    total = float(w.sum())
    if not np.isfinite(w).all() or w.min() < 0.0 or total <= _TINY:
        return np.full(N, 1.0 / N)
    return w / total


# ----------------------------------------------------------------------
# 面板装配：滚动协方差 → 求解器 → 权重面板（预热期回退等权）
# ----------------------------------------------------------------------

def _covariance_weights(data: MarketData, window: int,
                        solver: Callable[[np.ndarray], np.ndarray],
                        ridge: float = 1e-10) -> pd.DataFrame:
    """通用装配：逐日滚动样本协方差喂给 solver，产出 long-only 满仓权重面板。

    窗口不足、协方差非有限或 solver 输出异常时回退等权 1/N；正常行裁剪
    负值后归一，保证每行权重 ≥ 0 且和 = 1。
    """
    dates, symbols = data.dates, data.symbols
    N = max(len(symbols), 1)
    equal = np.full(N, 1.0 / N)
    vals = _returns_values(data)
    covs = _rolling_covariances(vals, window)
    out = np.tile(equal, (len(dates), 1))
    for t in range(len(dates)):
        S = covs[t]
        if not np.isfinite(S).all():
            continue
        w = np.clip(np.nan_to_num(solver(_regularize(S, ridge)),
                                  nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        total = float(w.sum())
        if total <= _TINY:
            continue
        out[t] = w / total
    return pd.DataFrame(out, index=dates, columns=symbols)


# ----------------------------------------------------------------------
# 策略
# ----------------------------------------------------------------------

class EqualWeightRebalanceStrategy(Strategy):
    """等权 + 定期再平衡（日历再平衡，买入持有漂移）。"""

    name = "equal_weight_rebal"
    channel = "allocation"
    universe = "cross_section"
    long_only = True
    description = "等权定期再平衡：每 k 期把组合调回 1/N 等权，其间买入持有让权重随收益自然漂移。"
    hypothesis = ("等权是不预测收益的分散化基准；日历再平衡隐含『卖高买低』，在均值回归/震荡市中"
                  "收割再平衡溢价并抑制权重漂移带来的集中度。在强者恒强的单边趋势市中，再平衡会"
                  "过早减持赢家而跑输纯买入持有。")
    source = "组合配置经典范式——等权组合 + 日历再平衡 (calendar rebalancing)，原创 numpy 实现。"
    params = {"rebal_freq": 21}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        k = max(int(self.params["rebal_freq"]), 1)
        vals = _returns_values(data)          # vals[t] 为 (t-1, t] 收益，t 时刻已知
        n, N = vals.shape
        N = max(N, 1)
        equal = np.full(N, 1.0 / N)
        out = np.empty((n, N))
        w = equal.copy()
        for t in range(n):
            if t % k == 0:
                w = equal.copy()              # 再平衡日：回到等权
            else:
                grown = np.clip(w * (1.0 + vals[t]), 0.0, None)   # 买入持有漂移
                total = float(grown.sum())
                w = grown / total if total > _TINY else equal.copy()
            out[t] = w
        return pd.DataFrame(out, index=data.dates, columns=data.symbols)


class InverseVolatilityStrategy(Strategy):
    """逆波动率加权（风险预算的简化形式，忽略相关性）。"""

    name = "inverse_vol"
    channel = "allocation"
    universe = "cross_section"
    long_only = True
    description = "逆波动率加权：权重 ∝ 1/滚动已实现波动（滚动样本协方差对角线），归一到每行和=1。"
    hypothesis = ("波动率的可预测性远强于收益率，按风险倒数分配资金可在不预测收益的前提下压低"
                  "组合波动、改善夏普。忽略相关性是其近似：当相关性结构剧变、低波资产拥挤或出现"
                  "尾部跳空时，配置会偏离真正的风险最优。")
    source = "组合配置经典范式——逆波动率加权 (inverse volatility weighting)，原创实现。"
    params = {"window": 60}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        return _covariance_weights(data, int(self.params["window"]),
                                   _inverse_vol_weights)


class RiskParityAllocationStrategy(Strategy):
    """风险平价 / 等风险贡献 (ERC)，自研阻尼定点迭代求解。"""

    name = "risk_parity_alloc"
    channel = "allocation"
    universe = "cross_section"
    long_only = True
    description = ("风险平价(ERC)：用滚动样本协方差，自研牛顿迭代（对数障碍凸形式 + 回溯线搜索）"
                   "使各资产成分风险贡献 wᵢ(Σw)ᵢ 相等，归一到每行和=1。")
    hypothesis = ("等风险贡献让组合风险在资产间均衡分散，而非被少数高波/高相关资产垄断，在风险"
                  "维度（而非资金维度）做分散化，长期风险调整后收益更稳健。当风险因子高度共振"
                  "（危机中相关性趋 1）时『分散化幻觉』破裂，可能出现同步回撤。")
    source = ("组合配置经典范式——Equal Risk Contribution / Risk Parity（Maillard-Roncalli-"
              "Teïletche 思想），牛顿迭代求解为本仓库原创纯 numpy 实现。")
    params = {"window": 60, "max_iters": 50, "tol": 1e-10}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        solver = partial(_risk_parity_weights, max_iters=int(p["max_iters"]),
                         tol=float(p["tol"]))
        return _covariance_weights(data, int(p["window"]), solver)


class MinVarianceAllocationStrategy(Strategy):
    """全局最小方差组合 (GMV)，Σ⁻¹1 + long-only 投影。"""

    name = "min_variance_alloc"
    channel = "allocation"
    universe = "cross_section"
    long_only = True
    description = ("全局最小方差：滚动样本协方差下解 w ∝ Σ⁻¹1（岭正则 + np.linalg.solve，奇异"
                   "回退 pinv），负分量裁剪到 0 后重新归一，保证 long-only 满仓。")
    hypothesis = ("协方差是收益分布中最可估计的部分，最小方差组合位于有效前沿最左端，在不预测"
                  "收益的前提下最小化组合波动。对协方差估计误差高度敏感（『误差最大化』效应），"
                  "样本病态时会给出极端权重——用岭正则与 long-only 投影约束之。")
    source = "组合配置经典范式——Markowitz 全局最小方差组合 (GMV)，纯 numpy 原创实现。"
    params = {"window": 60, "ridge": 1e-10}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        return _covariance_weights(data, int(p["window"]), _min_variance_weights,
                                   ridge=float(p["ridge"]))


class MaxDiversificationStrategy(Strategy):
    """最大分散化组合 (MDP)：最大化分散化比率，自研两阶段迭代求解。"""

    name = "max_diversification"
    channel = "allocation"
    universe = "cross_section"
    long_only = True
    description = ("最大分散化：最大化分散化比率 DR=(Σwᵢσᵢ)/σ_p，自研两阶段迭代（阻尼定点迭代 + "
                   "Frank-Wolfe 精确线搜索）求解，归一到每行和=1。")
    hypothesis = ("DR 度量『加权平均单资产波动』与『组合实际波动』之比，最大化 DR 即最大化分散化"
                  "收益，避免组合被相关性结构中暗含的共同风险主导，兼具低相关暴露与风险均衡。"
                  "当所有资产相关性趋同（系统性危机）时无处分散，DR 优势消失。")
    source = ("组合配置经典范式——Most Diversified Portfolio（Choueifaty-Coignard 分散化比率"
              "思想），两阶段迭代求解为本仓库原创纯 numpy 实现。")
    params = {"window": 60, "max_iters": 80, "gamma": 0.5, "tol": 1e-7,
              "fw_iters": 60, "fw_tol": 1e-9}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        solver = partial(_max_div_weights, max_iters=int(p["max_iters"]),
                         gamma=float(p["gamma"]), tol=float(p["tol"]),
                         fw_iters=int(p["fw_iters"]), fw_tol=float(p["fw_tol"]))
        return _covariance_weights(data, int(p["window"]), solver)
