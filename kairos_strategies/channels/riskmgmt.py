"""渠道：riskmgmt —— 组合层「风险管理 overlay」（在等权基准之上叠加风险控制）。

与既有渠道的分工：
  - ``allocation`` / ``taa`` 解决「钱投给谁、投多少」（选资产与战术配置）；
  - ``volatility`` 的 ``vol_target`` 用滚动标准差对等权组合做单一维度的波动缩放；
  - 本渠道把组合当作一个整体来做**下行风险控制**：基础敞口 = 等权组合 1/N，
    再叠加五类经典风控范式（组合保险 / 回撤节流 / 移动止损 / 波动目标 / 熔断冷却），
    输出组合层总敞口 e_t ∈ [0, 1]，均匀广播为每资产 e_t / N。

「现金」的表示方式（沿用 taa/volatility 渠道约定）：
  权重面板每行之和 ≤ 1，``1 - Σw`` 即现金比例；现金不占列、收益记 0。
  全部策略 long_only（权重 ≥ 0）、不加杠杆。

五个策略（均为本仓库原创实现，仅用 numpy/pandas）：
  1. ``cppi``                   固定比例组合保险：递推模拟受保护净值，
     敞口 = min(1, m×(V-floor)/V)，净值跌破 floor 后永久锁定空仓（吸收态）。
  2. ``drawdown_throttle``      回撤节流：敞口随「净值距历史高点的回撤」线性缩减，
     回撤达到 max_dd 时降至 0，回撤越深仓位越低。
  3. ``trailing_stop_overlay``  组合层移动止损：净值跌破 历史高点×(1-stop) 清仓，
     冻结高点参考，净值重新创出「冻结高点×(1+reentry)」的新高后才再入场（状态机）。
  4. ``portfolio_vol_target``   组合波动目标：按 目标波动 / 等权基准组合的 EWMA
     已实现波动 缩放总敞口（上限 1，余额现金）。与 ``vol_target``（滚动标准差）、
     ``vol_target_taa``（逆波动率 sleeve）在作用对象与波动估计上均不同。
  5. ``circuit_breaker``        熔断：滚动 window 期累计亏损触及 loss_threshold
     即空仓进入 cooldown 期冷却，冷却结束且亏损指标恢复后自动重新开仓。

防未来函数：第 t 期敞口只用截至 t（含 t 收盘）的等权基准净值/收益——历史高点为
expanding cummax、回撤与滚动亏损为历史窗口统计、波动为 EWMA 历史平滑；
CPPI / 移动止损 / 熔断是**路径依赖状态机**，严格按时间顺序递推，t 期决策只依赖
≤ t 的信息（策略自身净值也只由历史敞口 × 历史收益演化而来）。统计量不足的预热期
一律不建仓（全现金），绝不用未来数据回填。引擎还会再滞后一期，双重保险。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-12


# ----------------------------------------------------------------------
# 模块内自实现的小工具与状态机（不从其它渠道 import 私有函数）
# ----------------------------------------------------------------------

def _benchmark_returns(data: MarketData) -> pd.Series:
    """等权（每日再平衡）基准组合的单期收益率序列。"""
    return data.returns(1).mean(axis=1)


def _benchmark_nav(data: MarketData) -> pd.Series:
    """等权基准组合净值指数（起点 1.0，确定性），只用截至当期的收益。"""
    return (1.0 + _benchmark_returns(data)).cumprod()


def _broadcast(scale: pd.Series, data: MarketData) -> pd.DataFrame:
    """把组合层敞口序列广播成等权权重面板（每资产 scale_t / N，行和 = scale_t）。"""
    n = max(int(data.n_assets), 1)
    per = scale.reindex(data.dates).fillna(0.0).to_numpy(dtype=float) / n
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


def _ewma_vol(ret: pd.Series, span: int, periods_per_year: int) -> pd.Series:
    """EWMA 已实现波动（年化）：sqrt(EWMA(r²) × ppy)，样本不足 min_periods 则 NaN。"""
    sp = max(int(span), 2)
    var = ret.pow(2).ewm(span=sp, adjust=False, min_periods=sp).mean()
    return np.sqrt(var.clip(lower=0.0) * float(periods_per_year))


def _cppi_state_machine(rets: np.ndarray, floor: float, mult: float) -> np.ndarray:
    """CPPI 递推：按时间顺序模拟「受保护净值 V」并逐期给出敞口 e_t ∈ [0, 1]。

    V_0 = 1；e_t = clip(mult × (V_t - floor) / V_t, 0, 1)；
    V_{t} = V_{t-1} × (1 + e_{t-1} × r_t)（现金收益记 0，增长因子截断为非负）。
    一旦 V ≤ floor 则 e = 0，V 被冻结，此后**永久空仓**（吸收态，锁定保底语义）。
    t 期敞口只依赖 ≤ t 的收益，路径依赖、确定性。
    """
    n = len(rets)
    out = np.zeros(n, dtype=float)
    v = 1.0
    e_prev = 0.0
    for t in range(n):
        if t > 0:
            growth = max(1.0 + e_prev * float(rets[t]), 0.0)
            v = v * growth
        cushion = v - float(floor)
        e = 0.0 if cushion <= 0.0 else min(1.0, float(mult) * cushion / max(v, _EPS))
        out[t] = e
        e_prev = e
    return out


def _drawdown_exposure(nav: np.ndarray, max_dd: float) -> np.ndarray:
    """回撤节流：e_t = clip(1 - dd_t / max_dd, 0, 1)，dd 为净值距 expanding 历史高点的回撤。

    ``np.maximum.accumulate`` 只用到 ≤ t 的净值，无未来信息；回撤越大敞口越低，
    回撤达到 max_dd 时降至 0（线性节流）。
    """
    peak = np.maximum.accumulate(nav)
    dd = 1.0 - nav / np.maximum(peak, _EPS)
    return np.clip(1.0 - dd / max(float(max_dd), _EPS), 0.0, 1.0)


def _trailing_stop_state_machine(nav: np.ndarray, stop: float, reentry: float) -> np.ndarray:
    """组合层移动止损状态机（在场 / 场外两态）：

    - 在场：历史高点 peak 随净值创新高而上移；净值跌破 peak×(1-stop) → 清仓，
      并把 peak 冻结为离场参考 frozen；
    - 场外：不更新 peak；净值重新创出 frozen×(1+reentry) 的新高 → 再入场，
      并以当前净值为新的 peak 起点。
    t 期状态只用 ≤ t 的净值，离场当日即空仓（e_t = 0）。
    """
    n = len(nav)
    out = np.zeros(n, dtype=float)
    invested = True
    peak = float(nav[0])
    frozen = float(nav[0])
    for t in range(n):
        x = float(nav[t])
        if invested:
            if x > peak:
                peak = x
            if x < peak * (1.0 - float(stop)):
                invested = False
                frozen = peak
        else:
            if x > frozen * (1.0 + float(reentry)):
                invested = True
                peak = x
        out[t] = 1.0 if invested else 0.0
    return out


def _circuit_breaker_state_machine(nav: np.ndarray, window: int, threshold: float,
                                   cooldown: int) -> np.ndarray:
    """熔断状态机：滚动 window 期累计亏损 1 - NAV_t/NAV_{t-window} ≥ threshold 时熔断，

    熔断当日即空仓，并进入 cooldown 期冷却（其后连续 cooldown 期强制空仓）；
    冷却结束后若亏损指标已恢复（< threshold）则恢复满仓，否则再次熔断重新冷却。
    窗口不足（t < window）的预热期不建仓。t 期决策只用 ≤ t 的净值，路径依赖、确定性。
    """
    n = len(nav)
    w = max(int(window), 1)
    out = np.zeros(n, dtype=float)
    cool = 0
    for t in range(n):
        if cool > 0:                      # 冷却期内：强制空仓
            out[t] = 0.0
            cool -= 1
            continue
        if t < w:                         # 预热期：亏损指标未知 → 不建仓
            out[t] = 0.0
            continue
        loss = 1.0 - float(nav[t]) / max(float(nav[t - w]), _EPS)
        if loss >= float(threshold):      # 触及阈值：熔断当日空仓 + 进入冷却
            out[t] = 0.0
            cool = max(int(cooldown), 0)
        else:
            out[t] = 1.0
    return out


# ----------------------------------------------------------------------
# 策略 1：CPPI 固定比例组合保险
# ----------------------------------------------------------------------

class CppiStrategy(Strategy):
    """CPPI：等权基准之上叠加固定比例组合保险，净值跌破保底 floor 后永久空仓。"""

    name = "cppi"
    channel = "riskmgmt"
    universe = "overlay"
    long_only = True
    description = ("固定比例组合保险 (CPPI)：以等权组合为风险资产、现金为保本资产，递推模拟受保护"
                   "净值 V（起点 1，现金收益 0），每期敞口 e = min(1, multiplier × (V - floor) / V)，"
                   "剩余 1-e 配现金；V 跌破 floor 后敞口永久归零（吸收态），实现『最差亏到 floor 附近』"
                   "的保底语义。")
    hypothesis = ("核心假设：凸性保护（convex protection）——仓位随安全垫 (V-floor) 线性缩放，"
                  "让组合在接近保底线时自动去风险，理论上把最大回撤锁定在 1-floor 附近，代价是"
                  "『踏空风险』：崩盘后即便市场 V 型反转，策略也已锁定空仓无法参与（吸收态设计），"
                  "且缺口 (gap) 行情下实际净值可能击穿 floor（单期跌幅超过 1/multiplier 时安全垫"
                  "直接转负）。multiplier 越大越接近满仓持有、保护越弱；越小越保守、越容易永久出局。"
                  "适合有明确保底线需求的下行保护场景，不适合追求反弹收益的场景。")
    source = ("经典结构化产品范式——Constant Proportion Portfolio Insurance (Black-Perold / "
              "Black-Jones 的 CPPI)，此处为等权组合层的原创递推实现（策略自身净值状态机）。")
    params = {
        "floor": 0.8,          # 保底线：净值跌到 0.8（亏 20%）即永久空仓
        "multiplier": 5.0,     # 风险资产乘数 m：敞口 = min(1, m×(V-floor)/V)
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        rets = _benchmark_returns(data).to_numpy(dtype=float)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        e = _cppi_state_machine(rets, float(p["floor"]), float(p["multiplier"]))
        return _finalize(_broadcast(pd.Series(e, index=data.dates), data), data)


# ----------------------------------------------------------------------
# 策略 2：回撤节流
# ----------------------------------------------------------------------

class DrawdownThrottleStrategy(Strategy):
    """回撤节流：敞口随等权基准净值距历史高点的回撤线性缩减，深回撤降至空仓。"""

    name = "drawdown_throttle"
    channel = "riskmgmt"
    universe = "overlay"
    long_only = True
    description = ("回撤节流 overlay：实时计算等权基准净值距 expanding 历史高点的回撤 dd，"
                   "总敞口 e = clip(1 - dd / max_dd, 0, 1)——回撤越大仓位越低（线性节流），"
                   "回撤达到 max_dd 时降至全现金，净值收复回撤后敞口自动回升。")
    hypothesis = ("核心假设：回撤具有持续性（危机传播、去杠杆螺旋、趋势下行），『正在回撤』的组合"
                  "短期内更可能继续回撤，因此把回撤本身当作风险信号做连续降杠杆，可在深熊中显著"
                  "压低尾部损失；与二元止损相比，线性节流避免了在阈值附近的反复全进全出。"
                  "失效场景：V 型急跌急涨（节流在底部仓位最轻、反弹初期仍低仓，踏空最猛的一段）、"
                  "高波动震荡市（回撤信号频繁升降导致换手放大）；max_dd 越小越保守、对噪声越敏感。")
    source = ("风险管理实践——drawdown-based position sizing / 回撤控制 (drawdown control) 的"
              "组合层原创实现，等权基准 + 线性节流旋钮。")
    params = {
        "max_dd": 0.25,     # 回撤达到 25% 时敞口降为 0（线性节流的满刻度）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        nav = _benchmark_nav(data).to_numpy(dtype=float)
        e = _drawdown_exposure(nav, float(p["max_dd"]))
        return _finalize(_broadcast(pd.Series(e, index=data.dates), data), data)


# ----------------------------------------------------------------------
# 策略 3：组合层移动止损 overlay
# ----------------------------------------------------------------------

class TrailingStopOverlayStrategy(Strategy):
    """移动止损：净值跌破 历史高点×(1-stop) 清仓，重新创出冻结高点×(1+reentry) 新高再入场。"""

    name = "trailing_stop_overlay"
    channel = "riskmgmt"
    universe = "overlay"
    long_only = True
    description = ("组合层移动止损 overlay（两态状态机）：持有等权组合时跟踪净值历史高点，跌破"
                   "高点×(1-stop) 当日清仓并冻结高点为参考；场外等待净值重新创出 冻结高点×(1+reentry)"
                   " 的新高才再入场（入场后高点重新随净值上移）。止损-再入场之间全程现金。")
    hypothesis = ("核心假设：大幅回撤往往有惯性（熊市趋势），截断左尾的价值高于放弃震荡市的小幅"
                  "收益；『必须创出带缓冲的新高才回来』的再入场条件把 V 型假反弹与真正的趋势修复"
                  "区分开，避免止损后在下跌中继里反复接飞刀。失效场景：宽幅震荡市（止损线附近"
                  "反复触发、每次都是低点卖出不回来）、以及止损后立刻 V 型反转并快速创新高的行情"
                  "（reentry 缓冲要求先收复全部失地再涨 reentry，踏空整段修复）；stop 越小越敏感、"
                  "误触发越多，reentry 越大越难回来。")
    source = ("经典风控范式——trailing stop / 跟踪止损的组合层状态机原创实现（含离场冻结高点 +"
              "新高再入场的滞后带设计）。")
    params = {
        "stop": 0.10,       # 止损线：净值跌破历史高点 10% 清仓
        "reentry": 0.02,    # 再入场缓冲：须创出冻结高点 ×1.02 的新高
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        nav = _benchmark_nav(data).to_numpy(dtype=float)
        e = _trailing_stop_state_machine(nav, float(p["stop"]), float(p["reentry"]))
        return _finalize(_broadcast(pd.Series(e, index=data.dates), data), data)


# ----------------------------------------------------------------------
# 策略 4：组合波动目标（等权基准层）
# ----------------------------------------------------------------------

class PortfolioVolTargetStrategy(Strategy):
    """组合波动目标：按 目标波动 / 等权基准组合 EWMA 已实现波动 缩放总敞口。"""

    name = "portfolio_vol_target"
    channel = "riskmgmt"
    universe = "overlay"
    long_only = True
    description = ("组合波动目标 overlay：对『等权 1/N 基准组合』本身做风险预算——用 EWMA"
                   "（RiskMetrics 风格，平方收益指数平滑）估计基准组合的年化已实现波动，"
                   "总敞口 e = clip(target_vol / 已实现波动, 0, 1)，每资产 e/N，余额留现金；"
                   "波动未知的预热期不建仓。")
    hypothesis = ("核心假设：波动率聚集且可预测（EWMA 对突变反应快于等权滚动窗口），高波动期"
                  "单位风险补偿更差，把组合波动锁定在目标水平能压低下行尾部并改善风险调整后收益。"
                  "与既有策略的区别：``vol_target`` 用滚动标准差估计、``vol_target_taa`` 作用于"
                  "逆波动率 sleeve，本策略直接对等权基准组合做 EWMA 波动目标，是『组合层风险预算』"
                  "的最小实现。失效场景：目标波动长期高于已实现波动时退化为满仓持有（上限 1 不加"
                  "杠杆）；波动骤升后急速回落的 V 型行情因平滑滞后而踏空；上行加速伴随高波动时被"
                  "系统性降仓。")
    source = ("风险管理范式——volatility targeting / RiskMetrics EWMA 波动估计在等权基准组合层"
              "的原创实现（与本仓库 vol_target、vol_target_taa 在作用对象与估计方法上均不同）。")
    params = {
        "target_vol": 0.10,    # 年化目标组合波动
        "span": 20,            # EWMA 平滑参数（span）
        "vol_floor": 0.01,     # 年化波动下限，防止近零波动把敞口顶爆
        "max_scale": 1.0,      # 敞口上限：1 = 不加杠杆
        "min_scale": 0.0,      # 敞口下限：0 = 允许完全转现金
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        rv = _ewma_vol(_benchmark_returns(data), int(p["span"]), data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        scale = (float(p["target_vol"]) / rv).clip(lower=float(p["min_scale"]),
                                                   upper=float(p["max_scale"]))
        scale = scale.reindex(data.dates).fillna(0.0)      # 波动未知（预热）→ 不建仓
        return _finalize(_broadcast(scale, data), data)


# ----------------------------------------------------------------------
# 策略 5：熔断（滚动亏损触发 + 冷却期）
# ----------------------------------------------------------------------

class CircuitBreakerStrategy(Strategy):
    """熔断：滚动窗口累计亏损触及阈值即空仓冷却 N 期，冷却结束且指标恢复后重新开仓。"""

    name = "circuit_breaker"
    channel = "riskmgmt"
    universe = "overlay"
    long_only = True
    description = ("组合层熔断 overlay：监控等权基准净值的滚动 window 期累计亏损 "
                   "1 - NAV_t/NAV_{t-window}，一旦 ≥ loss_threshold 立即熔断——当日空仓并进入"
                   " cooldown 期强制冷却（全程现金）；冷却结束后若亏损指标已回落到阈值之下则恢复"
                   "满仓等权，否则再次熔断重新冷却。窗口不足的预热期不建仓。")
    hypothesis = ("核心假设：短期内的急速下跌（恐慌抛售、流动性螺旋）往往需要时间消化，暴跌后"
                  "立即接回的风险远大于等待——『强制冷静期』把交易者的择时问题变成规则问题，避免在"
                  "下跌未止时反复抄底。与移动止损的区别：熔断的退出条件是**时间**（冷却期）而非"
                  "价格（新高），因此恢复更快、代价是可能在下行中继里过早接回（若指标仍超阈值则"
                  "自动再熔断，形成滚动保护）。失效场景：单日 gap 崩盘（滚动窗口内首末日对比可能"
                  "低估或高估损失）、V 型反转（冷却期踏空反弹起点）；阈值越小/窗口越短越敏感。")
    source = ("交易所熔断机制 (circuit breaker) 的组合层移植——滚动亏损触发 + 定时冷却的原创"
              "状态机实现。")
    params = {
        "window": 10,             # 滚动亏损窗口（期）
        "loss_threshold": 0.08,   # 触发阈值：窗口内累计亏损 ≥ 8% 熔断
        "cooldown": 21,           # 冷却期长度（期），期间强制空仓
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        nav = _benchmark_nav(data).to_numpy(dtype=float)
        e = _circuit_breaker_state_machine(nav, int(p["window"]),
                                           float(p["loss_threshold"]), int(p["cooldown"]))
        return _finalize(_broadcast(pd.Series(e, index=data.dates), data), data)


# 便于外部（报告/研究记录）按渠道枚举的辅助常量
CHANNEL = "riskmgmt"
STRATEGY_NAMES = (
    "cppi",
    "drawdown_throttle",
    "trailing_stop_overlay",
    "portfolio_vol_target",
    "circuit_breaker",
)

__all__ = [
    "CppiStrategy",
    "DrawdownThrottleStrategy",
    "TrailingStopOverlayStrategy",
    "PortfolioVolTargetStrategy",
    "CircuitBreakerStrategy",
    "CHANNEL",
    "STRATEGY_NAMES",
]
