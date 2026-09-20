"""渠道：hrp —— 层次/聚类组合配置策略（cross_section，只做多，满仓）。

收集途径：层次聚类配置范式——层次风险平价 (HRP)、层次等风险贡献 (HERC)、
聚类逆波动率。全部为本仓库原创实现：相关/协方差用 ``data.returns()`` 的
滚动样本窗口估计；层次聚类优先使用 ``scipy.cluster.hierarchy.linkage``
（single-linkage），**scipy 为可选依赖**——运行时 import 失败则自动回退到
自研 numpy 凝聚聚类（同为 single-linkage，argmin 按行主序取首个最小值对，
对距离并列有确定性打破规则）；树的子节点次序按「最小叶索引在前」排序、
扁平簇按簇创建 id 升序输出，全链路确定性。

HRP 四步范式（López de Prado 思想的原创 numpy 实现）：
  ① 相关矩阵 → 距离矩阵 ``d = sqrt(0.5·(1-ρ))``（ρ∈[-1,1] ⇒ d∈[0,1]）；
  ② 层次聚类：对距离矩阵做 single-linkage 凝聚聚类，得到二叉树状图；
  ③ 准对角化：按树状图叶序重排协方差矩阵，使相似资产在矩阵中相邻；
  ④ 递归二分：从根节点起二分，两个子簇的预算按「簇内组合方差」倒数
     分配 ``α_L = 1 - σ²_L/(σ²_L + σ²_R)``（簇方差用簇内逆波动率组合
     方差度量），风险越低的簇分到越多资金，直到叶子得到最终权重。

统一范式（防未来函数）：第 t 期权重只用截至 t 的收益率窗口
``[t-window+1, t]`` 估计相关/协方差（vals[t] 为 (t-1, t] 收益，t 时刻
已知，不再额外 shift）；窗口不足（t < window-1）回退等权 1/N。引擎还会
再滞后一期，双重保险。所有策略 long_only（权重 ≥ 0）且逐行归一：每行
权重和 ≈ 1（满仓配置）。
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_TINY = 1e-18


# ----------------------------------------------------------------------
# 通用小工具：收益率矩阵 / 滚动协方差 / 相关→距离 / 正则化
# ----------------------------------------------------------------------

def _returns_values(data: MarketData) -> np.ndarray:
    """收益率面板 → 有限 numpy 矩阵（NaN/inf 置 0；价格恒正故收益 > -1）。"""
    vals = data.returns().to_numpy(dtype=float)
    vals = np.atleast_2d(vals)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _regularize(S: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """对称化 + 岭正则：Σ_reg = Σ + ridge·mean(diag Σ)·I，保证数值正定。

    样本协方差在资产近似共线（如高相关簇）时会病态，微小岭项既稳定后续的
    方差/迭代计算，又几乎不改变估计本身；迹为 0（全零收益）时用绝对下限，
    保证退化窗口下相关/聚类/权重仍有确定性定义。
    """
    S = 0.5 * (S + S.T)
    N = S.shape[0]
    scale = float(np.trace(S)) / max(N, 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return S + max(ridge * scale, _TINY) * np.eye(N)


def _corr_from_cov(S: np.ndarray) -> np.ndarray:
    """协方差 → 相关矩阵：零方差资产对他资产相关置 0、自相关置 1（确定性）。"""
    sd = np.sqrt(np.clip(np.diag(S), 0.0, None))
    ok = sd > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, sd, 1.0), 0.0)
    C = S * np.outer(inv, inv)
    C = np.clip(np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def _distance_from_corr(C: np.ndarray) -> np.ndarray:
    """HRP 第①步：相关 → 距离 ``d = sqrt(0.5·(1-ρ))`` ∈ [0,1]，对角为 0。"""
    D = np.sqrt(np.clip(0.5 * (1.0 - C), 0.0, 1.0))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# ----------------------------------------------------------------------
# 层次聚类（HRP 第②步）：scipy 优先，缺失时 numpy 自研凝聚聚类回退
# ----------------------------------------------------------------------

def _scipy_linkage(D: np.ndarray) -> Optional[np.ndarray]:
    """尝试 scipy single-linkage，返回 linkage 矩阵；scipy 缺失/失败返回 None。

    每次调用时现场 import（不缓存模块对象），使得把 ``sys.modules['scipy']``
    置 None 即可触发 numpy 回退路径；任何异常都吞掉并回退，保证离线可用。
    """
    try:
        import scipy  # noqa: F401  先探测顶层包：sys.modules['scipy']=None 时在此抛错
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return None
    try:
        condensed = squareform(np.nan_to_num(D, nan=1.0), checks=False)
        Z = linkage(condensed, method="single")
        return np.asarray(Z, dtype=float)
    except Exception:
        return None


def _agglomerative_merges(D: np.ndarray) -> List[Tuple[int, int]]:
    """自研 numpy 凝聚聚类（single-linkage），scipy 缺失时的确定性回退。

    维护 (2N-1)² 的簇间距离表（未激活项为 inf），每步取当前活跃簇对中的
    最小距离合并；**并列打破规则**：对对称矩阵按行主序取首个最小值对，即
    (行 id 最小, 列 id 最小)，完全确定。新簇距离 = min(两父簇距离)
    （single-linkage/最近邻），合并序列与 scipy linkage 语义一致：
    第 s 次合并产生新簇 id = N+s，返回 [(a_0,b_0), ..., (a_{N-2},b_{N-2})]。
    """
    N = int(D.shape[0])
    if N <= 1:
        return []
    M = 2 * N - 1
    cd = np.full((M, M), np.inf)
    iu = np.triu_indices(N, 1)
    cd[iu] = np.nan_to_num(D, nan=1.0)[iu]
    cd = np.minimum(cd, cd.T)                    # 对称化（inf 安全）
    np.fill_diagonal(cd, np.inf)
    active = np.zeros(M, dtype=bool)
    active[:N] = True
    merges: List[Tuple[int, int]] = []
    for s in range(N - 1):
        work = np.where(active[:, None] & active[None, :], cd, np.inf)
        flat = int(np.argmin(work))              # 行主序首个最小 → 确定性
        a, b = divmod(flat, M)
        new = N + s
        d_new = np.minimum(cd[a, :], cd[b, :])
        cd[new, :] = d_new
        cd[:, new] = d_new
        cd[new, new] = np.inf
        cd[new, a] = cd[a, new] = np.inf
        cd[new, b] = cd[b, new] = np.inf
        active[a] = active[b] = False
        active[new] = True
        merges.append((a, b))
    return merges


def _linkage_merges(D: np.ndarray) -> List[Tuple[int, int]]:
    """距离矩阵 → 合并序列：优先 scipy linkage，缺失时 numpy 自研回退。"""
    N = int(D.shape[0])
    if N <= 1:
        return []
    Z = _scipy_linkage(D)
    if Z is not None and Z.shape[0] == N - 1:
        return [(int(Z[i, 0]), int(Z[i, 1])) for i in range(N - 1)]
    return _agglomerative_merges(D)


# ----------------------------------------------------------------------
# 树工具：二叉树重建 / 叶序（准对角化，HRP 第③步）/ 扁平簇切分
# ----------------------------------------------------------------------

def _build_children(merges: List[Tuple[int, int]], N: int) -> List[Optional[Tuple[int, int]]]:
    """合并序列 → children 表（长度 2N-1；叶子为 None）。

    子节点次序确定性规则：按「最小叶索引升序」排列左右孩子。
    """
    total = 2 * N - 1
    children: List[Optional[Tuple[int, int]]] = [None] * total
    min_leaf = list(range(N)) + [0] * (N - 1)
    for s, (a, b) in enumerate(merges):
        nid = N + s
        if min_leaf[b] < min_leaf[a]:
            a, b = b, a
        children[nid] = (a, b)
        min_leaf[nid] = min_leaf[a]
    return children


def _leaf_order(children: List[Optional[Tuple[int, int]]], root: int) -> List[int]:
    """树状图叶序（迭代式中序遍历，左=最小叶索引小的一侧），用于准对角化。"""
    order: List[int] = []
    stack = [root]
    while stack:
        nid = stack.pop()
        ch = children[nid]
        if ch is None:
            order.append(nid)
        else:
            stack.append(ch[1])
            stack.append(ch[0])
    return order


def _flat_clusters(merges: List[Tuple[int, int]], N: int, k: int) -> List[List[int]]:
    """在树状图上切出 k 个扁平簇：只施加前 N-k 次合并，剩余连通分量即簇。

    输出按簇创建 id 升序、成员升序，确定性；k 会被裁剪到 [1, N]。
    """
    k = int(min(max(int(k), 1), N))
    members: Dict[int, List[int]] = {i: [i] for i in range(N)}
    consumed = set()
    for s in range(min(N - k, len(merges))):
        a, b = merges[s]
        nid = N + s
        members[nid] = members[a] + members[b]
        consumed.add(a)
        consumed.add(b)
    return [sorted(members[i]) for i in sorted(members) if i not in consumed]


# ----------------------------------------------------------------------
# 权重求解器：HRP 递归二分 / 簇内 ERC / 簇内逆波动率
# ----------------------------------------------------------------------

def _cluster_variance(Sqd: np.ndarray, lo: int, hi: int) -> float:
    """准对角化协方差的连续切片 [lo,hi) 的簇方差：簇内逆波动率组合的方差。"""
    sub = Sqd[lo:hi, lo:hi]
    sd = np.sqrt(np.clip(np.diag(sub), _TINY, None))
    cw = 1.0 / sd
    cw = cw / float(cw.sum())
    v = float(cw @ (sub @ cw))
    return v if np.isfinite(v) and v > 0.0 else 0.0


def _hrp_weights(S: np.ndarray, merges: List[Tuple[int, int]]) -> np.ndarray:
    """HRP 第③④步：准对角化 + 递归二分，返回 long-only、和=1 的权重。

    ③ 按叶序重排协方差 Σ_qd = Σ[order, order]——树中每个子树的叶子在
       叶序下必为连续切片，故簇方差可直接取 Σ_qd 的连续子块；
    ④ 从根递归二分：左右子簇预算按 ``α_L = 1 - σ²_L/(σ²_L+σ²_R)``
       分配（方差低的簇多分），并列/退化时 α=0.5；到叶子即最终权重。
       每层 α + (1-α) = 1 ⇒ 权重和恒为 1，且天然 ≥ 0。
    """
    N = int(S.shape[0])
    if N == 1:
        return np.ones(1)
    root = N + len(merges) - 1
    children = _build_children(merges, N)
    order = _leaf_order(children, root)
    Sqd = S[np.ix_(order, order)]              # ③ 准对角化
    pos = np.empty(N, dtype=int)
    for p, leaf in enumerate(order):
        pos[leaf] = p
    lo = np.empty(2 * N - 1, dtype=int)
    hi = np.empty(2 * N - 1, dtype=int)
    for nid in range(N):
        lo[nid], hi[nid] = pos[nid], pos[nid] + 1
    for nid in range(N, 2 * N - 1):
        ch = children[nid]
        if ch is not None:                     # 子树切片 = 孩子切片的并（连续）
            lo[nid] = min(lo[ch[0]], lo[ch[1]])
            hi[nid] = max(hi[ch[0]], hi[ch[1]])
    w = np.zeros(N)
    stack: List[Tuple[int, float]] = [(root, 1.0)]
    while stack:                               # ④ 递归二分
        nid, budget = stack.pop()
        ch = children[nid]
        if ch is None:
            w[nid] = budget
            continue
        l, r = ch
        vl = _cluster_variance(Sqd, lo[l], hi[l])
        vr = _cluster_variance(Sqd, lo[r], hi[r])
        tot = vl + vr
        alpha = 0.5 if tot <= _TINY else float(np.clip(1.0 - vl / tot, 0.0, 1.0))
        stack.append((l, budget * alpha))
        stack.append((r, budget * (1.0 - alpha)))
    return w


def _iv_weights(S: np.ndarray) -> np.ndarray:
    """逆波动率权重：w ∝ 1/σ（σ 取协方差对角线开方），归一到和=1。"""
    sd = np.sqrt(np.clip(np.diag(S), _TINY, None))
    w = 1.0 / sd
    return w / float(w.sum())


def _erc_weights(S: np.ndarray, max_iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """簇内等风险贡献 (ERC) 权重：自研阻尼乘法定点迭代。

    目标：各资产成分风险贡献 ``RC_i = w_i·(Σw)_i`` 相等（预算 1/N）。
    一阶定点为 ``w_i ∝ b_i/(Σw)_i``，迭代 ``w ← w ⊙ sqrt(r/geomean(r))``
    （r_i = b_i/(Σw)_i，除以几何均值做尺度中心化，归一化吸收统一缩放），
    指数 0.5 阻尼保证对良态正定 Σ 稳定收敛；对数域运算天然保持 w > 0
    （long-only）。不收敛/数值异常时回退逆波动率（同为风险均衡的近似）。
    """
    N = int(S.shape[0])
    if N == 1:
        return np.ones(1)
    fallback = _iv_weights(S)
    w = fallback.copy()
    budget = 1.0 / N
    for _ in range(max(max_iters, 1)):
        y = np.maximum(S @ w, _TINY)           # 正定 Σ、w>0 ⇒ y>0；保底裁剪
        r = np.clip(budget / y, _TINY, None)
        geo = float(np.exp(np.mean(np.log(r))))
        if not np.isfinite(geo) or geo <= 0.0:
            break
        rel = r / geo
        w_new = w * np.sqrt(rel)
        total = float(w_new.sum())
        if not np.isfinite(w_new).all() or total <= _TINY:
            break
        w = w_new / total
        if float(np.max(np.abs(rel - 1.0))) < tol:
            break
    total = float(w.sum())
    if not np.isfinite(w).all() or w.min() < 0.0 or total <= _TINY:
        return fallback
    return w / total


def _herc_weights(S: np.ndarray, merges: List[Tuple[int, int]],
                  k: int = 2, erc_iters: int = 200, erc_tol: float = 1e-12) -> np.ndarray:
    """HERC：树状图切 k 簇，簇间等权（各 1/k），簇内 ERC 均衡风险贡献。"""
    N = int(S.shape[0])
    if N == 1:
        return np.ones(1)
    clusters = _flat_clusters(merges, N, k)
    share = 1.0 / max(len(clusters), 1)
    w = np.zeros(N)
    for mem in clusters:
        idx = np.asarray(mem, dtype=int)
        sub = S[np.ix_(idx, idx)]
        w[idx] = share * _erc_weights(sub, max_iters=erc_iters, tol=erc_tol)
    return w


def _clustered_iv_weights(S: np.ndarray, merges: List[Tuple[int, int]],
                          k: int = 2) -> np.ndarray:
    """聚类逆波动率：树状图切 k 簇，簇内逆波动率加权、簇间等权。"""
    N = int(S.shape[0])
    if N == 1:
        return np.ones(1)
    clusters = _flat_clusters(merges, N, k)
    share = 1.0 / max(len(clusters), 1)
    w = np.zeros(N)
    for mem in clusters:
        idx = np.asarray(mem, dtype=int)
        sub = S[np.ix_(idx, idx)]
        w[idx] = share * _iv_weights(sub)
    return w


# ----------------------------------------------------------------------
# 面板装配：滚动相关/协方差 → 聚类 → 求解器 → 权重面板（预热期回退等权）
# ----------------------------------------------------------------------

def _hierarchical_weights(data: MarketData, window: int,
                          allocator: Callable[[np.ndarray, List[Tuple[int, int]]],
                                              np.ndarray]) -> pd.DataFrame:
    """通用装配：逐日用截至 t 的滚动窗口估计相关/协方差，聚类后喂给 allocator。

    窗口不足、协方差非有限、树结构异常或 allocator 输出异常时回退等权
    1/N；正常行裁剪负值后归一，保证每行权重 ≥ 0 且和 = 1。
    """
    dates, symbols = data.dates, data.symbols
    n = len(dates)
    vals = _returns_values(data)
    N = int(vals.shape[1])
    window = max(int(window), 2)
    equal = np.full(N, 1.0 / N) if N > 0 else np.zeros(0)
    out = np.tile(equal, (n, 1))
    for t in range(window - 1, n):
        block = vals[t - window + 1: t + 1]    # 只用截至 t 的信息（防未来）
        S = np.atleast_2d(np.cov(block, rowvar=False, ddof=1))
        if S.shape != (N, N) or not np.isfinite(S).all():
            continue
        S = _regularize(S)
        D = _distance_from_corr(_corr_from_cov(S))     # ① 相关 → 距离
        merges = _linkage_merges(D)                    # ② 层次聚类
        if N > 1 and len(merges) != N - 1:
            continue
        w = np.clip(np.nan_to_num(allocator(S, merges),
                                  nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        total = float(w.sum())
        if not np.isfinite(total) or total <= _TINY:
            continue
        out[t] = w / total
    return pd.DataFrame(out, index=dates, columns=symbols)


# ----------------------------------------------------------------------
# 策略
# ----------------------------------------------------------------------

class HierarchicalRiskParityStrategy(Strategy):
    """层次风险平价 (HRP)：聚类树准对角化 + 递归二分按簇方差倒数分预算。"""

    name = "hrp"
    channel = "hrp"
    universe = "cross_section"
    long_only = True
    description = ("层次风险平价(HRP)：①相关矩阵→距离 sqrt(0.5(1-ρ))；②层次聚类（scipy "
                   "single-linkage，缺失回退自研 numpy 凝聚聚类）；③按树状图叶序重排协方差"
                   "（准对角化）；④递归二分，按簇内逆波动组合方差倒数 α=1-σ²_L/(σ²_L+σ²_R) "
                   "分配预算，归一到每行和=1。")
    hypothesis = ("把『相关结构聚类』与『风险预算二分』结合，无需矩阵求逆即可利用协方差信息，"
                  "对估计误差远不如最小方差敏感，在高相关/病态样本下仍给出分散的 long-only "
                  "配置。当相关性结构快速切换（危机中簇边界失稳）或窗口过短导致聚类抖动时，"
                  "权重换手上升、分散化效果打折。")
    source = ("层次配置经典范式——Hierarchical Risk Parity（López de Prado 四步思想的原创"
              "实现）；聚类用 scipy linkage，缺失时自研 numpy 凝聚聚类回退，全链路确定性。")
    params = {"window": 60}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        return _hierarchical_weights(data, int(self.params["window"]), _hrp_weights)


class HierarchicalEqualRiskContributionStrategy(Strategy):
    """层次等风险贡献 (HERC)：树切 k 簇，簇间等权、簇内 ERC 均衡。"""

    name = "herc"
    channel = "hrp"
    universe = "cross_section"
    long_only = True
    description = ("层次等风险贡献(HERC)：滚动相关距离→层次聚类后在树状图上切 k 簇；簇间等权"
                   "（各 1/k），簇内用自研阻尼乘法定点迭代求等风险贡献(ERC)权重，使簇内各资产"
                   "风险贡献 wᵢ(Σw)ᵢ 相等，归一到每行和=1。")
    hypothesis = ("『先聚类分预算、再簇内均衡风险』把风险平价约束在相关子结构内部，避免全局 "
                  "ERC 被跨簇共同风险主导；簇间等权不依赖簇数估计的精度，配置更均衡稳健。当"
                  "簇数 k 与真实相关结构错配、或簇内资产数过少时，退化为近似逆波动率配置。")
    source = ("层次配置经典范式——Hierarchical Equal Risk Contribution（Pozzi 等 HERC 思想"
              "的原创实现）；簇内 ERC 用自研乘法定点迭代，scipy 缺失时聚类自动 numpy 回退。")
    params = {"window": 60, "n_clusters": 2, "erc_iters": 200, "erc_tol": 1e-12}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        allocator = partial(_herc_weights, k=int(p["n_clusters"]),
                            erc_iters=int(p["erc_iters"]), erc_tol=float(p["erc_tol"]))
        return _hierarchical_weights(data, int(p["window"]), allocator)


class ClusteredInverseVolatilityStrategy(Strategy):
    """聚类逆波动率：树切 k 簇，簇内逆波动率加权、簇间等权。"""

    name = "clustered_inverse_vol"
    channel = "hrp"
    universe = "cross_section"
    long_only = True
    description = ("聚类逆波动率：滚动相关距离→层次聚类切 k 簇；簇内按 1/σ 逆波动率加权，簇间"
                   "等权（各 1/k），归一到每行和=1。相比全局逆波动率，先隔离相关子结构再在簇内"
                   "比较波动，避免高相关资产叠加重仓。")
    hypothesis = ("普通逆波动率忽略相关性：一簇高度相关的高波资产仍可能合计吃掉大部分资金。"
                  "簇间等权把预算先均分到相关子结构，簇内再按波动倒数细分，兼得相关性分散与"
                  "波动率缩放。当聚类边界在窗口间抖动时换手上升；k 过大时退化为全局逆波动率。")
    source = ("层次配置经典范式——clustered inverse volatility（聚类 + 逆波动率的原创组合"
              "实现）；scipy 缺失时聚类自动回退自研 numpy 凝聚算法，确定性打破并列。")
    params = {"window": 60, "n_clusters": 2}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        allocator = partial(_clustered_iv_weights, k=int(p["n_clusters"]))
        return _hierarchical_weights(data, int(p["window"]), allocator)
