"""渠道：adaptive —— 在线学习 / 专家加权元策略（随时间自适应的序贯决策）。

与 ensemble 渠道的区别（务必分清）：ensemble 是**静态/滚动统计加权**——等权、
逆波动、滚动夏普等系数只依赖历史收益的低阶统计量，规则本身不随奖励路径演化；
本渠道是**在线学习 (online learning)**——把本仓库其它渠道的公开 Strategy 类当作
「专家 (experts)」，用序贯决策算法在奖励流上不断更新内部状态（累计增益、拉动
计数、相对优势、波动状态），第 t 期的配置是**整条奖励历史的函数**，会随市场
regime 变化而自行漂移、切换甚至改变自身参数：

  hedge_experts          Hedge/指数权重（Freund-Littlestone-Long 自研版）：
                         w_b[t] ∝ exp(η·G_b[t-1])，G 为专家累计对数收益（可折扣）。
  bandit_select          多臂老虎机：滑动窗口 UCB1 式打分（窗口均值 + 拉动计数
                         探索项），每 hold 期确定性选中一个专家整段持有。
  online_mom_rev_switch  在线动量/回归切换：并行跟踪动量专家与均值回归专家的
                         相对优势 z 值，逐期把权重连续地偏向当前更优的一类。
  adaptive_param         在线自适应参数：均线择时的窗口长度与敞口随「截至 t-1」
                         的已实现波动在线调整（高波 → 更短窗口 + 更低敞口）。

固定专家池（全部 import 其它渠道的**公开** Strategy 类，不 import 任何私有下划线
函数；被 import 的渠道都不 import adaptive，因此无循环依赖）：
  technical.SmaCrossStrategy           (sma_cross)         双均线趋势择时
  technical.DonchianTurtleStrategy     (donchian_turtle)   通道突破趋势择时
  technical.RsiReversionStrategy       (rsi_reversion)     RSI 超卖回归择时
  momentum.TsMomentumStrategy          (ts_momentum)       时序动量择时
  trend.TsMomVolScaled                 (tsmom_volscaled)   动量 + 波动率目标
  factor.LowVolatilityStrategy         (low_volatility)    低波动截面异象
  allocation.InverseVolatilityStrategy (inverse_vol)       逆波动率截面配置

防未来函数（本渠道的核心工程约束，逐条对应在线学习的「只用 ≤t-1 奖励」）：
  1. 专家面板本身逐期只用截至当期的价格（由各来源渠道保证，且经前缀不变性测试）。
  2. 专家「已实现奖励」用 ``engine.Backtester`` 零成本回测：
     ``r_b[t] = Σ_i w_b[t-1, i] · (p[t, i]/p[t-1, i] - 1)``，t 期期末才可观测。
  3. 在线状态严格滞后一期：Hedge 的累计增益做 ``shift(1)``（第 t 期权重 =
     softmax(η·G[≤t-1])）；bandit 的窗口均值只用行 [t-window, t)（即 ≤t-1 的
     奖励）且仅在换段时刻重估；切换器的滚动均值/标准差先算后整体 ``shift(1)``；
     adaptive_param 的窗口与敞口由 ``shift(1)`` 后的已实现波动决定。
     因此第 t 期的一切系数/统计/参数只依赖 ≤t-1 的已实现信息，绝不偷看当期与未来。
  4. 预热期（奖励观测不足）回退等权/中性配置，不引入任何前视。

long_only 与预算约束：专家面板先经 ``align_weights(long_only=True)`` 统一为
[0,1] 且 NaN→0；各在线算法的专家系数逐行非负且和 ≤ 1（Hedge/bandit/切换器
恰为 1），故每行权重和 = Σ_b coef_b·(专家面板行和) ≤ 1；末尾再做一次防御性
去杠杆（仅当行和 > 1+1e-12 时按比例缩放）。adaptive_param 单资产权重 =
信号(0/1)×敞口(≤1)/N，逐行和天然 ≤ 1。

确定性：全部算法为确定性数值规则——softmax/累计增益无随机；bandit 用
``np.argmax``（并列取专家池顺序中靠前者），无 RNG；同一 MarketData 多次调用
结果逐位一致。

性能：对同一份 MarketData，专家面板与专家回测奖励只计算一次（模块内 LRU 缓存，
键为价格面板指纹，纯性能优化、不改变输出）；在线递推为 O(n·k) 的 numpy/标量
混合循环，1000×8 数据上四个策略合计在数秒内。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy, align_weights
from ..engine import Backtester
from .allocation import InverseVolatilityStrategy
from .factor import LowVolatilityStrategy
from .momentum import TsMomentumStrategy
from .technical import DonchianTurtleStrategy, RsiReversionStrategy, SmaCrossStrategy
from .trend import TsMomVolScaled

_TINY = 1e-18          # 归一化除零保护
_LEV_TOL = 1e-12       # 防御性去杠杆触发容差
_R_FLOOR = -0.999      # 对数收益的下限保护（价格恒正 ⇒ r > -1，再留安全边际）
_CACHE_LIMIT = 6       # 专家面板/奖励缓存条目上限（LRU）

#: 在线学习共用的专家池（顺序即系数矩阵列顺序 / bandit 并列打破顺序，确定性）
EXPERT_BASES: Tuple[Tuple[str, Type[Strategy]], ...] = (
    ("sma_cross", SmaCrossStrategy),
    ("donchian_turtle", DonchianTurtleStrategy),
    ("rsi_reversion", RsiReversionStrategy),
    ("ts_momentum", TsMomentumStrategy),
    ("tsmom_volscaled", TsMomVolScaled),
    ("low_volatility", LowVolatilityStrategy),
    ("inverse_vol", InverseVolatilityStrategy),
)

#: online_mom_rev_switch 的两条腿（均已在专家池内，直接复用其面板与奖励）
MOMENTUM_LEG = "ts_momentum"
REVERSION_LEG = "rsi_reversion"

_CACHE: "OrderedDict[tuple, Tuple[Dict[str, pd.DataFrame], pd.DataFrame]]" = OrderedDict()


# ----------------------------------------------------------------------
# 专家池：公开访问器（供测试与本渠道内部复用）
# ----------------------------------------------------------------------

def expert_base_names() -> List[str]:
    """专家池名称（固定顺序，即系数矩阵列顺序）。"""
    return [nm for nm, _ in EXPERT_BASES]


def expert_strategies() -> "OrderedDict[str, Strategy]":
    """实例化专家池全部基础策略，键为策略 name。"""
    out: "OrderedDict[str, Strategy]" = OrderedDict()
    for nm, cls in EXPERT_BASES:
        out[nm] = cls()
    return out


def switch_leg_names() -> Tuple[str, str]:
    """online_mom_rev_switch 的两条腿名：(动量腿, 均值回归腿)。"""
    return MOMENTUM_LEG, REVERSION_LEG


def expert_weight_panels(data: MarketData,
                         names: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """专家的 long_only 权重面板表（副本，可安全修改）。

    键为专家 name，值为 index=data.dates / columns=data.symbols 的权重面板，
    已统一裁剪到 [0, 1] 并把 NaN 填 0。
    """
    panels, _ = _snapshot(data)
    keys = list(names) if names is not None else expert_base_names()
    return {nm: panels[nm].copy() for nm in keys}


def expert_backtest_returns(data: MarketData,
                            names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """专家的零成本回测（毛）收益面板（副本），即在线学习的「奖励流」。

    index=data.dates，columns=专家名；第 t 行为该专家在 (t-1, t] 区间的已实现
    组合收益（由 ``Backtester`` 用「上一期权重 × 本期资产收益」计算，t 期期末
    才可知——因此第 t 期决策最多只能用到第 t-1 行，见各算法的 shift/切片约定）。
    """
    _, rets = _snapshot(data)
    if names is None:
        return rets.copy()
    return rets[list(names)].copy()


# ----------------------------------------------------------------------
# 内部工具：指纹缓存 / 面板统一 / 线性混合
# ----------------------------------------------------------------------

def _fingerprint(data: MarketData) -> tuple:
    """MarketData 的确定性指纹（价格内容 + 形状 + 索引端点 + 年化期数）。"""
    p = data.prices
    vals = np.ascontiguousarray(p.to_numpy(dtype="float64"))
    idx = p.index
    head = str(idx[0]) if len(idx) else ""
    tail = str(idx[-1]) if len(idx) else ""
    vol_shape = None if data.volumes is None else tuple(data.volumes.shape)
    return (vals.shape, tuple(str(c) for c in p.columns), head, tail, vol_shape,
            int(data.periods_per_year), hash(vals.tobytes()))


def _long_only(w: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """把任意专家权重统一成对齐 data、值域 [0, 1]、无 NaN 的 long_only 面板。"""
    return align_weights(w, data, long_only=True, clip=1.0)


def _snapshot(data: MarketData) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """计算（或命中缓存）专家面板表与专家回测奖励面板。

    缓存是纯性能优化：键含价格面板全部字节的哈希，命中即数据完全相同，
    因此不改变任何输出（离线、确定性）。缓存值在模块内一律只读。
    """
    key = _fingerprint(data)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    panels: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for nm, cls in EXPERT_BASES:
        panels[nm] = _long_only(cls().generate_weights(data), data)

    bt = Backtester(cost_rate=0.0, risk_free=0.0,
                    periods_per_year=int(data.periods_per_year))
    cols: "OrderedDict[str, pd.Series]" = OrderedDict()
    for nm in expert_base_names():
        res = bt.run(data, panels[nm])
        cols[nm] = res.gross_returns.reindex(data.dates).fillna(0.0)
    rets = pd.DataFrame(cols, index=data.dates)
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    _CACHE[key] = (panels, rets)
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return panels, rets


def _reward_values(rets: pd.DataFrame) -> np.ndarray:
    """奖励面板 → 有限 numpy 矩阵（NaN/inf 置 0），形状 (n_days, k_experts)。"""
    vals = rets.to_numpy(dtype="float64")
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _blend_panels(panels: Sequence[pd.DataFrame], coef: np.ndarray,
                  data: MarketData) -> pd.DataFrame:
    """按逐期专家系数线性混合专家权重面板，并做防御性去杠杆（每行和 ≤ 1）。"""
    n = len(data.dates)
    m = len(data.symbols)
    acc = np.zeros((n, m), dtype="float64")
    for j, panel in enumerate(panels):
        acc += coef[:, j:j + 1] * panel.reindex(index=data.dates,
                                                columns=data.symbols).to_numpy(dtype="float64")
    acc = np.clip(acc, 0.0, None)
    total = acc.sum(axis=1)
    scale = np.where(total > 1.0 + _LEV_TOL, total, 1.0)
    acc = acc / scale[:, None]
    return pd.DataFrame(acc, index=data.dates, columns=data.symbols)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、去 NaN、裁到 [0, cap]，行和超过 cap 时按比例缩回（兜底）。"""
    out = w.reindex(index=data.dates, columns=data.symbols)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=cap)
    total = out.sum(axis=1)
    factor = (cap / total.replace(0.0, np.nan)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


# ----------------------------------------------------------------------
# 在线学习核心递推（全部确定性、只用 ≤t-1 的奖励）
# ----------------------------------------------------------------------

def _hedge_coefficients(rets: pd.DataFrame, eta: float, decay: float) -> pd.DataFrame:
    """Hedge/指数权重（Freund-Littlestone-Long 的自研实现，含可选折扣）。

    逐步递推：g_b[t] = log(1 + r_b[t])（对数奖励），
      G_b[t] = decay·G_b[t-1] + g_b[t]   （decay=1 即经典 FLL 累计增益；
                                            decay<1 为折扣 Hedge，记忆随时间指数
                                            衰减，对非平稳 regime 反应更快）
      w_b[t] = exp(η·G_b[t-1]) / Σ_b' exp(η·G_b'[t-1])
    第 t 期权重只用 G[≤t-1]（实现上把 G 整体下移一行），t=0 无历史 ⇒ 等权。
    softmax 先减行最大值再取指数（数值稳定，不改变结果）。逐行非负、和恰为 1。
    """
    vals = np.clip(_reward_values(rets), _R_FLOOR, None)
    g = np.log1p(vals)
    n, k = g.shape
    d = float(np.clip(decay, 0.0, 1.0))
    eta = float(eta)
    G = np.empty((n, max(k, 1)), dtype="float64")
    acc = np.zeros(max(k, 1), dtype="float64")
    for t in range(n):
        acc = acc * d + g[t]
        G[t] = acc
    prev = np.zeros_like(G)                 # 防未来：第 t 期系数只用 G[t-1]
    if n > 1:
        prev[1:] = G[:-1]
    z = eta * prev
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    coef = e / np.maximum(e.sum(axis=1, keepdims=True), _TINY)
    return pd.DataFrame(coef, index=rets.index, columns=rets.columns)


def _bandit_coefficients(rets: pd.DataFrame, window: int, hold: int,
                         explore_c: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """滑动窗口 UCB1 式多臂老虎机（确定性，无 RNG）。

    把时间分成长度 hold 的「段」：段首时刻 t（t ≥ warmup=window，且
    (t-warmup) mod hold == 0）用**只含 ≤t-1 奖励**的滑动窗口 [t-window, t) 打分：
      score_b = mean(r_b[t-window .. t-1]) + c·sqrt(2·ln(N_blk) / max(n_b, 1))
    其中 n_b 为该专家历史被选中的段数（拉动计数）、N_blk 为已完成的段数——
    第二项即 UCB1 的探索奖励：被拉动越少的专家分数加成越大，防止过早锁死在
    早期赢家上。选中 score 最高者（``np.argmax``，并列取池内靠前者，确定性），
    整段持有其权重面板（系数 one-hot）。预热期（t < window，奖励观测不足）
    回退 1/k 等权混合。返回 (系数面板, 逐期选中专家下标数组；预热期为 -1)。
    """
    vals = _reward_values(rets)
    n, k = vals.shape
    k = max(int(k), 1)
    window = max(int(window), 2)
    hold = max(int(hold), 1)
    c = max(float(explore_c), 0.0)
    coef = np.full((n, k), 1.0 / k, dtype="float64")
    sel = np.full(n, -1, dtype=np.int64)
    pulls = np.zeros(k, dtype="float64")
    blocks = 0
    current = -1
    for t in range(n):
        if t < window:
            continue                                   # 预热期：等权（防未来地不做选择）
        if current < 0 or (t - window) % hold == 0:      # 段首重估（只用 [t-window, t) 的奖励）
            mu = vals[t - window:t].mean(axis=0)
            nb = max(blocks, 1)
            bonus = c * np.sqrt(2.0 * np.log(nb) / np.maximum(pulls, 1.0))
            current = int(np.argmax(mu + bonus))         # 并列取首个最大下标，确定性
            pulls[current] += 1.0
            blocks += 1
        sel[t] = current
        coef[t] = 0.0
        coef[t, current] = 1.0
    return pd.DataFrame(coef, index=rets.index, columns=rets.columns), sel


def _switch_coefficients(diff: pd.Series, window: int, tilt_max: float,
                         std_floor: float) -> pd.DataFrame:
    """在线动量/回归切换系数：相对优势的有界软倾斜（确定性）。

    输入 diff = r_mom - r_rev（两专家逐期已实现奖励差）。逐期计算滚动窗口均值
    与标准差后**整体 shift(1)**：z[t] = mean(diff[t-window..t-1]) /
    max(std(同窗), floor)——第 t 期只用 ≤t-1 的奖励差。倾斜量
    tilt = tilt_max·z/(1+|z|) ∈ (-tilt_max, tilt_max)（有界、单调、无需指数），
    w_mom = 0.5 + tilt、w_rev = 0.5 - tilt：相对优势越大权重越偏向该腿，
    但任何单期都不把另一腿压到 0.5-tilt_max 以下（保留在线纠错能力）。
    预热期（观测不足 / z 为 NaN）回退中性 0.5/0.5。逐行非负、和恰为 1。
    """
    window = max(int(window), 2)
    tilt_max = float(np.clip(tilt_max, 0.0, 0.5))
    floor = max(float(std_floor), _TINY)
    roll = diff.rolling(window, min_periods=window)
    z = (roll.mean() / roll.std(ddof=0).clip(lower=floor)).shift(1)   # 防未来：滞后一期
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype="float64")
    tilt = tilt_max * z / (1.0 + np.abs(z))
    out = np.stack([0.5 + tilt, 0.5 - tilt], axis=1)
    return pd.DataFrame(out, index=diff.index, columns=["momentum", "reversion"])


# ----------------------------------------------------------------------
# 元策略公共基类（name 保持 'base'，registry 不会收录）
# ----------------------------------------------------------------------

class _AdaptiveMetaStrategy(Strategy):
    """adaptive 元策略公共基类：准备专家面板并做线性混合。"""

    universe = "cross_section"
    long_only = True

    def _expert_panels(self, data: MarketData) -> List[pd.DataFrame]:
        panels, _ = _snapshot(data)
        return [panels[nm] for nm in expert_base_names()]


# ----------------------------------------------------------------------
# 策略 1：Hedge / 指数权重
# ----------------------------------------------------------------------

class HedgeExpertsStrategy(_AdaptiveMetaStrategy):
    """Hedge 指数权重在线学习元策略：w_b[t] ∝ exp(η·累计对数奖励[t-1])。"""

    name = "hedge_experts"
    channel = "adaptive"
    universe = "cross_section"
    long_only = True
    description = ("Hedge/指数权重在线学习元策略（Freund-Littlestone-Long 自研版）：先用 "
                   "engine.Backtester 对 sma_cross、donchian_turtle、rsi_reversion、ts_momentum、"
                   "tsmom_volscaled、low_volatility、inverse_vol 七个专家各做一次零成本回测得到奖励"
                   "序列，维护每个专家的累计对数收益 G_b（可选 decay<1 折扣旧奖励），第 t 期按 "
                   "softmax(η·G_b[t-1]) 归一出专家权重并混合其权重面板；η 学习率可调，t=0 等权。")
    hypothesis = ("核心假设：专家表现存在可被奖励路径追踪的持续性——指数权重把资金按 exp(η·累计"
                  "对数收益) 倾斜给历史赢家，是乘法权重更新 (MWU) 的经典在线学习算法，对『池中存在"
                  "长期优秀专家』的情形有遗憾界保证（相对最优单专家的差距随时间收敛）；η 越大对"
                  "近期表现越敏感、追随越快但噪声越大，decay<1 引入遗忘因子使其在非平稳 regime "
                  "切换中更快改押新赢家。防未来：G 整体下移一行，第 t 期权重严格只用 ≤t-1 的已实现"
                  "奖励。失效场景：专家收益强均值回复（赢家随即变输家）时指数加权系统性追高杀低；"
                  "η 过大时权重被单期极端奖励主导而剧烈抖动，抬高换手；全部专家同步亏损时它只能"
                  "『矮子里拔将军』，无法降低总敞口。")
    source = ("在线学习经典算法——Hedge / 指数权重（Freund-Littlestone-Long 的 multiplicative "
              "weights 思想）与折扣 Hedge 变体的自研实现；专家池复用本仓库 technical / momentum / "
              "trend / factor / allocation 渠道的公开 Strategy 类，奖励由 engine.Backtester 回测产生。")
    params = {"eta": 8.0, "decay": 1.0, "cost_rate": 0.0,
              "n_bases": len(EXPERT_BASES), "bases": expert_base_names()}

    def mix_coefficients(self, data: MarketData) -> pd.DataFrame:
        """逐期 Hedge 专家权重：index=data.dates，columns=专家名。

        逐行非负、和恰为 1；第 t 行只用专家在 [0, t-1] 的回测奖励。
        """
        _, rets = _snapshot(data)
        return _hedge_coefficients(rets, eta=float(self.params["eta"]),
                                   decay=float(self.params["decay"]))

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        panels = self._expert_panels(data)
        coef = self.mix_coefficients(data)[expert_base_names()].to_numpy(dtype="float64")
        return _blend_panels(panels, coef, data)


# ----------------------------------------------------------------------
# 策略 2：多臂老虎机（滑动窗口 UCB1 式，段内持有单一专家）
# ----------------------------------------------------------------------

class BanditSelectStrategy(_AdaptiveMetaStrategy):
    """多臂老虎机在线选择元策略：每段整仓持有窗口 UCB 分数最高的专家。"""

    name = "bandit_select"
    channel = "adaptive"
    universe = "cross_section"
    long_only = True
    description = ("多臂老虎机在线选择元策略：把 sma_cross、donchian_turtle、rsi_reversion、"
                   "ts_momentum、tsmom_volscaled、low_volatility、inverse_vol 七个专家当作臂，"
                   "每 hold 期为一个段，段首用只含 ≤t-1 奖励的滑动窗口 [t-window, t) 计算 "
                   "UCB1 式分数 = 窗口均值 + explore_c·sqrt(2·ln(段数)/拉动次数)，确定性选中"
                   "（argmax，并列取池内靠前者）分数最高的专家整段持有其权重面板；预热期等权。")
    hypothesis = ("核心假设：与 Hedge 的连续加权不同，bandit 做**离散切换**——每次只押注当前"
                  "统计意义上最优的单一专家，在市场 regime 分明（某类专家显著占优）时比软加权"
                  "更锐利、跟得上切换；UCB1 的探索项按拉动次数衰减，保证被冷落的专家仍会被"
                  "周期性复查，避免在统计噪声上过早锁死（乐观面对不确定性）。滑动窗口均值使"
                  "统计量只反映近期表现，适应非平稳奖励。防未来：打分只用 [t-window, t) 的"
                  "已实现奖励，段内不换仓。失效场景：专家表现接近时离散切换制造高换手（每次"
                  "换臂都是全组合调仓）；窗口过短则均值估计噪声主导、频繁误切换；探索项在"
                  "专家数少时可能反复拉起真正的差专家。")
    source = ("在线学习经典算法——多臂老虎机 UCB1（Auer-Cesa-Bianchi-Fischer 思想）的滑动窗口"
              "确定性变体自研实现；专家池复用本仓库其它渠道公开 Strategy 类，奖励由 "
              "engine.Backtester 回测产生，无 RNG、并列按池序打破。")
    params = {"window": 63, "hold": 21, "explore_c": 0.0002, "cost_rate": 0.0,
              "n_bases": len(EXPERT_BASES), "bases": expert_base_names()}

    def mix_coefficients(self, data: MarketData) -> pd.DataFrame:
        """逐期 one-hot/等权系数面板：index=data.dates，columns=专家名。

        预热期（前 window 行）每行 1/k 等权；其后每行恰有一个 1（选中的专家）。
        第 t 行只用专家在 [0, t-1] 的回测奖励。
        """
        coef, _ = _bandit_coefficients(self._rewards(data),
                                       window=int(self.params["window"]),
                                       hold=int(self.params["hold"]),
                                       explore_c=float(self.params["explore_c"]))
        return coef

    def selections(self, data: MarketData) -> pd.Series:
        """逐期选中的专家名（object Series）；预热期为空串 ''。"""
        _, sel = _bandit_coefficients(self._rewards(data),
                                      window=int(self.params["window"]),
                                      hold=int(self.params["hold"]),
                                      explore_c=float(self.params["explore_c"]))
        names = expert_base_names()
        labels = [names[i] if i >= 0 else "" for i in sel]
        return pd.Series(labels, index=data.dates, name="selection", dtype="object")

    def _rewards(self, data: MarketData) -> pd.DataFrame:
        _, rets = _snapshot(data)
        return rets[expert_base_names()]

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        panels = self._expert_panels(data)
        coef = self.mix_coefficients(data).to_numpy(dtype="float64")
        return _blend_panels(panels, coef, data)


# ----------------------------------------------------------------------
# 策略 3：在线动量/均值回归切换
# ----------------------------------------------------------------------

class OnlineMomRevSwitchStrategy(_AdaptiveMetaStrategy):
    """在线风格切换元策略：按两腿相对优势把权重连续偏向动量或均值回归。"""

    name = "online_mom_rev_switch"
    channel = "adaptive"
    universe = "cross_section"
    long_only = True
    description = ("在线动量/均值回归切换元策略：并行跟踪动量专家 ts_momentum 与均值回归专家 "
                   "rsi_reversion 的零成本回测奖励，第 t 期用「截至 t-1」的 window 期奖励差均值/"
                   "标准差得到相对优势 z 值，倾斜量 tilt = tilt_max·z/(1+|z|)，按 (0.5+tilt, "
                   "0.5-tilt) 在线混合两腿权重面板；预热期或优势不明时回退中性 50/50。")
    hypothesis = ("核心假设：市场在『趋势 regime』与『震荡 regime』之间交替——动量专家在单边"
                  "行情赚趋势钱、在震荡市被反复止损；均值回归专家恰好相反。两类奖励的滚动相对"
                  "优势是 regime 的可观测代理，把权重连续地偏向近期更优的一类可让元策略净值"
                  "自适应两种行情（在线的风格轮动）。有界软倾斜 z/(1+|z|) 保证任一腿权重不低于 "
                  "0.5-tilt_max：即使判断错误也保留另一腿的纠错仓位，避免硬切换的双倍打脸。"
                  "防未来：滚动均值/标准差算完后整体 shift(1)，第 t 期系数严格只用 ≤t-1 的已实现"
                  "奖励差。失效场景：regime 快速交替（窗口内两种行情各半）时 z 值在 0 附近抖动、"
                  "切换滞后于 regime 本身；两腿表现高度相关时相对优势不含信息，退化为近似 50/50。")
    source = ("在线学习范式——专家跟踪 (expert tracking) / 在线 regime 切换的自研实现：对两个"
              "对立风格专家维护滚动相对优势统计量并做有界软倾斜；两腿分别复用本仓库 "
              "momentum.TsMomentumStrategy 与 technical.RsiReversionStrategy 公开类，"
              "奖励由 engine.Backtester 回测产生。")
    params = {"window": 42, "tilt_max": 0.35, "std_floor": 1e-4,
              "momentum_leg": MOMENTUM_LEG, "reversion_leg": REVERSION_LEG,
              "cost_rate": 0.0}

    def leg_coefficients(self, data: MarketData) -> pd.DataFrame:
        """逐期两腿系数：index=data.dates，columns=['momentum', 'reversion']。

        逐行非负、和恰为 1、每列 ∈ [0.5-tilt_max, 0.5+tilt_max]；第 t 行只用
        两腿在 [0, t-1] 的回测奖励差。
        """
        _, rets = _snapshot(data)
        mom, rev = switch_leg_names()
        diff = (rets[mom] - rets[rev]).rename("diff")
        return _switch_coefficients(diff, window=int(self.params["window"]),
                                    tilt_max=float(self.params["tilt_max"]),
                                    std_floor=float(self.params["std_floor"]))

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        panels, _ = _snapshot(data)
        mom, rev = switch_leg_names()
        coef = self.leg_coefficients(data).to_numpy(dtype="float64")
        return _blend_panels([panels[mom], panels[rev]], coef, data)


# ----------------------------------------------------------------------
# 策略 4：在线自适应参数（窗口/敞口随已实现波动在线调整）
# ----------------------------------------------------------------------

class AdaptiveParamStrategy(Strategy):
    """自适应参数择时：均线窗口与敞口随「截至 t-1」的已实现波动在线调整。"""

    name = "adaptive_param"
    channel = "adaptive"
    universe = "timing"
    long_only = True
    description = ("在线自适应参数择时（模块内自实现均线逻辑，不改其它渠道）：以「价格上穿均线"
                   "做多」为基础逻辑，但均线窗口不再固定——第 t 期用截至 t-1 的 vol_window 期"
                   "年化已实现波动 σ_{t-1} 在线设定有效窗口 W_t = clip(round(base_window·"
                   "ref_vol/σ_{t-1}), min_window, max_window)（高波 → 更短窗口、更快响应），"
                   "敞口同缩放为 clip(target_vol/σ_{t-1}, 0, 1)（高波 → 更低敞口），单资产权重 = "
                   "信号(0/1)×敞口/N，逐行和 ≤ 1。变窗口均线用前缀和按格点查表，全向量化。")
    hypothesis = ("核心假设：最优平滑窗口与波动 regime 相关——高波动期价格信息更快被噪声稀释，"
                  "更短的窗口能更快跟上趋势转折；同时波动率目标化的敞口把单位时间风险拉平"
                  "（高波自动降低敞口、低波放大到上限 1）。两者都是对『参数应随环境在线调整』"
                  "这一自适应思想的直接实现。防未来：σ 面板整体 shift(1)，第 t 期的窗口与敞口"
                  "参数严格由 ≤t-1 的已实现收益决定；价格与均线比较只用截至 t 的收盘价"
                  "（引擎还会再滞后一期）。失效场景：波动率跳变时参数调整滞后一期，窗口在"
                  "临界值附近抖动会放大换手；低波动长趋势中窗口触顶 max_window，均线过于迟钝。")
    source = ("在线自适应参数 (adaptive parameter / volatility-scaled window) 范式——固定规则"
              "的参数随已实现波动状态在线调整；均线择时骨架参考经典趋势跟随，变窗口前缀和实现、"
              "参数映射与敞口缩放为本渠道 100% 原创实现（不复用、不修改其它渠道代码）。")
    params = {"base_window": 30, "min_window": 8, "max_window": 80,
              "vol_window": 21, "ref_vol": 0.20, "target_vol": 0.15,
              "vol_floor": 0.02}

    # ---- 在线状态：σ_{t-1} 面板与由其决定的参数面板 ----

    def _lagged_sigma(self, data: MarketData) -> pd.DataFrame:
        """截至 t-1 的年化已实现波动（shift(1) 后），预热期回退 ref_vol。"""
        p = self.params
        close = data.prices.astype("float64")
        rets = close.pct_change().fillna(0.0)
        vw = max(int(p["vol_window"]), 2)
        rv = (rets.rolling(vw, min_periods=vw).std()
              * np.sqrt(float(data.periods_per_year))).shift(1)   # 防未来：只用 ≤t-1
        ref = float(p["ref_vol"])
        out = rv.replace([np.inf, -np.inf], np.nan).fillna(ref)
        return out.clip(lower=float(p["vol_floor"]))

    def effective_windows(self, data: MarketData) -> pd.DataFrame:
        """第 t 期各资产的有效均线窗口（整数面板，由 σ_{t-1} 在线决定）。"""
        p = self.params
        sigma = self._lagged_sigma(data)
        w = (float(p["base_window"]) * float(p["ref_vol"]) / sigma).round()
        return w.clip(lower=int(p["min_window"]), upper=int(p["max_window"])).astype(int)

    def exposure_scale(self, data: MarketData) -> pd.DataFrame:
        """第 t 期各资产的敞口缩放 ∈ [0, 1]（由 σ_{t-1} 在线决定，高波更低）。"""
        p = self.params
        sigma = self._lagged_sigma(data)
        return (float(p["target_vol"]) / sigma).clip(lower=0.0, upper=1.0)

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        vals = np.nan_to_num(close.to_numpy(dtype="float64"), nan=0.0,
                             posinf=0.0, neginf=0.0)
        n, m = vals.shape
        win = self.effective_windows(data).to_numpy()          # (n, m) 整数窗口，≤t-1 决定
        expo = self.exposure_scale(data).to_numpy(dtype="float64")

        # 变窗口均线：前缀和查表 ma[t] = (cs[t+1] - cs[t+1-W]) / W（窗口不足则无信号）
        cs = np.vstack([np.zeros((1, m), dtype="float64"), np.cumsum(vals, axis=0)])
        rows = np.broadcast_to(np.arange(1, n + 1, dtype=np.int64)[:, None], (n, m))
        idx = rows - win
        valid = idx >= 0
        prev = np.take_along_axis(cs, np.clip(idx, 0, None), axis=0)
        ma = (cs[1:] - prev) / np.maximum(win, 1).astype("float64")

        signal = valid & (vals > ma)
        w = signal.astype("float64") * expo / max(int(data.n_assets), 1)
        return _finalize(pd.DataFrame(w, index=data.dates, columns=data.symbols), data)
