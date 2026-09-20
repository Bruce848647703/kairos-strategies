"""渠道：ensemble —— 元策略（strategy of strategies，"策略的策略"）。

收集途径：组合管理层面的「策略池」范式（fund-of-strategies / manager momentum /
risk-based strategy allocation）。本渠道不发明新的行情信号，而是把本仓库**已稳定**
渠道里的公开 Strategy 类当作基础资产，先各自产出目标权重面板，再在**策略层**做加权
混合，输出一个单一的元策略权重面板。

固定基础策略池（全部 import 其它渠道的公开类，不 import 任何私有下划线函数；
被 import 的渠道都不 import ensemble，因此不存在循环依赖）：
  technical.SmaCrossStrategy        (sma_cross)         双均线趋势择时
  technical.DonchianTurtleStrategy  (donchian_turtle)   通道突破趋势择时
  momentum.TsMomentumStrategy       (ts_momentum)       时序动量择时
  factor.LowVolatilityStrategy      (low_volatility)    低波动截面异象
  allocation.InverseVolatilityStrategy (inverse_vol)    逆波动率截面配置
  meanrev.ZscoreReversionStrategy   (zscore_reversion)  z-score 均值回归（仅供混合腿）

四个元策略：
  equal_weight_ensemble    基础面板 1/k 等权平均（不预测哪个策略接下来有效）。
  inverse_vol_ensemble     按基础策略「截至 t-1」的滚动回测波动倒数加权（walk-forward）。
  sharpe_weighted_ensemble 按基础策略「截至 t-1」的滚动夏普(clip>=0)加权（walk-forward）。
  trend_reversion_blend    趋势腿(donchian_turtle) 60% + 均值回归腿(zscore_reversion 多头) 40%。

防未来函数（本渠道的核心工程约束）：
  1. 基础面板本身逐期只用截至当期的价格（由各渠道保证），且经前缀不变性测试验证。
  2. 基础策略的「历史表现」用 ``engine.Backtester`` 零成本回测得到收益序列
     ``r_b[t] = Σ_i w_b[t-1, i] · (p[t, i]/p[t-1, i] - 1)``，即 t 期期末已知的已实现收益。
  3. 混合系数在第 t 期只用 ``r_b[0 .. t-1]``：滚动统计量（波动/夏普）算完后再整体
     ``shift(1)``，因此 t 期权重绝不依赖 t 期及之后的收益，更不依赖未来价格。
  4. 预热期（有效观测 < min_obs）或系数退化（全部为 0）时回退等权，不引入任何前视。

long_only 与预算约束：所有基础面板先经 ``align_weights(long_only=True)`` 统一为
[0, 1] 且 NaN→0（可多空的基础腿因此只保留其多头部分）；混合系数逐行非负且和为 1，
故元策略每行权重和 = Σ_b coef_b · (基础面板行和) ≤ 1，无杠杆；末尾再做一次防御性
去杠杆（仅当行和 > 1 + 1e-12 时按比例缩放），保证数值误差下也满足约束。

universe 统一标注为 'cross_section'：元策略输出的是跨资产的组合级配置（每行和≈1），
尽管其基础腿里既有逐资产择时(timing)也有截面(cross_section)策略。

性能：对同一份 MarketData，基础面板与基础回测收益只计算一次（模块内小型 LRU 缓存，
键为价格面板指纹，纯确定性、离线）；混合是 k 次矩阵乘加，1000×8 数据上四个元策略
合计仍在 1 秒量级。
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
from .meanrev import ZscoreReversionStrategy
from .momentum import TsMomentumStrategy
from .technical import DonchianTurtleStrategy, SmaCrossStrategy

_TINY = 1e-18          # 系数归一的除零保护
_LEV_TOL = 1e-12       # 防御性去杠杆的触发容差（避免对合法行做无谓缩放）
_CACHE_LIMIT = 6       # 基础面板/回测缓存条目上限（LRU）

#: 加权类元策略共用的基础策略池（顺序即系数矩阵列顺序，确定性）
CORE_BASES: Tuple[Tuple[str, Type[Strategy]], ...] = (
    ("sma_cross", SmaCrossStrategy),
    ("donchian_turtle", DonchianTurtleStrategy),
    ("ts_momentum", TsMomentumStrategy),
    ("low_volatility", LowVolatilityStrategy),
    ("inverse_vol", InverseVolatilityStrategy),
)

#: 仅供 trend_reversion_blend 使用的均值回归腿（可多空，混合时只取多头）
REVERSION_BASES: Tuple[Tuple[str, Type[Strategy]], ...] = (
    ("zscore_reversion", ZscoreReversionStrategy),
)

_CACHE: "OrderedDict[tuple, Tuple[Dict[str, pd.DataFrame], pd.DataFrame]]" = OrderedDict()


# ----------------------------------------------------------------------
# 基础策略池：公开访问器（供测试与本渠道内部复用）
# ----------------------------------------------------------------------

def core_base_names() -> List[str]:
    """加权类元策略使用的基础策略名（固定顺序）。"""
    return [nm for nm, _ in CORE_BASES]


def base_strategies() -> "OrderedDict[str, Strategy]":
    """实例化本渠道用到的全部基础策略（核心池 + 回归腿），键为策略 name。"""
    out: "OrderedDict[str, Strategy]" = OrderedDict()
    for nm, cls in CORE_BASES + REVERSION_BASES:
        out[nm] = cls()
    return out


def base_weight_panels(data: MarketData,
                       names: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """基础策略的 long_only 权重面板表（副本，可安全修改）。

    键为策略 name，值为 index=data.dates / columns=data.symbols 的权重面板，
    已统一裁剪到 [0, 1] 并把 NaN 填 0（可多空基础腿只保留多头部分）。
    """
    panels, _ = _snapshot(data)
    keys = list(names) if names is not None else list(panels.keys())
    return {nm: panels[nm].copy() for nm in keys}


def base_backtest_returns(data: MarketData,
                          names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """基础策略的零成本回测（毛）收益面板（副本）。

    index=data.dates，columns=核心基础策略名；第 t 行为该基础策略在 (t-1, t] 区间
    的已实现组合收益（由 ``Backtester`` 用「上一期权重 × 本期资产收益」计算，t 期期末已知）。
    """
    _, rets = _snapshot(data)
    if names is None:
        return rets.copy()
    return rets[list(names)].copy()


# ----------------------------------------------------------------------
# 内部工具：指纹缓存 / long_only 统一 / 混合系数 / 线性混合
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
    """把任意基础策略权重统一成对齐 data、值域 [0, 1]、无 NaN 的 long_only 面板。"""
    return align_weights(w, data, long_only=True, clip=1.0)


def _snapshot(data: MarketData) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """计算（或命中缓存）基础面板表与基础策略回测收益面板。

    缓存是纯性能优化：键含价格面板全部字节的哈希，命中即数据完全相同，
    因此不改变任何输出（离线、确定性）。
    """
    key = _fingerprint(data)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    strats = base_strategies()
    panels: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for nm, s in strats.items():
        panels[nm] = _long_only(s.generate_weights(data), data)

    bt = Backtester(cost_rate=0.0, risk_free=0.0,
                    periods_per_year=int(data.periods_per_year))
    cols: "OrderedDict[str, pd.Series]" = OrderedDict()
    for nm in core_base_names():
        res = bt.run(data, panels[nm])
        cols[nm] = res.gross_returns.reindex(data.dates).fillna(0.0)
    rets = pd.DataFrame(cols, index=data.dates)
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    _CACHE[key] = (panels, rets)
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return panels, rets


def _coefficients(rets: pd.DataFrame, kind: str, window: int, min_obs: int,
                  vol_floor: float, periods_per_year: int) -> pd.DataFrame:
    """由基础策略收益面板计算逐期混合系数（防未来的关键一步）。

    kind='inverse_vol'：coef ∝ 1 / 滚动波动（波动越低权重越高）。
    kind='sharpe'     ：coef ∝ clip(滚动年化夏普, 0, ∞)（负夏普得 0 权重）。

    滚动统计只用截至当期的收益窗口，随后整体 ``shift(1)``：第 t 期系数严格由
    ``rets[0 .. t-1]`` 决定。逐行归一到和为 1；预热期（观测不足）或整行退化
    （全 0/全 NaN）回退等权 1/k，保证系数恒非负且和≈1。
    """
    if kind not in ("inverse_vol", "sharpe"):
        raise ValueError(f"未知的加权方式: {kind}")
    k = rets.shape[1]
    window = max(int(window), 2)
    min_obs = int(np.clip(int(min_obs), 2, window))
    floor = max(float(vol_floor), _TINY)

    roll = rets.rolling(window, min_periods=min_obs)
    sd = roll.std(ddof=1)
    if kind == "sharpe":
        mu = roll.mean()
        stat = (mu / sd.where(sd > floor)) * np.sqrt(float(periods_per_year))
        raw = stat.clip(lower=0.0)
    else:
        raw = (1.0 / sd.clip(lower=floor)).clip(lower=0.0)

    raw = raw.shift(1).replace([np.inf, -np.inf], np.nan)      # 防未来：整体滞后一期
    vals = np.nan_to_num(raw.to_numpy(dtype="float64"),
                         nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.clip(vals, 0.0, None)
    total = vals.sum(axis=1)
    bad = total <= _TINY
    denom = np.where(bad, 1.0, total)[:, None]
    equal = 1.0 / float(max(k, 1))
    out = np.where(bad[:, None], equal, vals / denom)
    return pd.DataFrame(out, index=rets.index, columns=rets.columns)


def _blend(panels: Sequence[pd.DataFrame], coef: np.ndarray,
           data: MarketData) -> pd.DataFrame:
    """按逐期系数线性混合基础权重面板，并做防御性去杠杆（每行和 ≤ 1）。"""
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


def _constant_coefficients(data: MarketData, k: int) -> np.ndarray:
    """等权系数矩阵 (n, k)，每行 1/k。"""
    return np.full((len(data.dates), max(int(k), 1)), 1.0 / float(max(int(k), 1)))


# ----------------------------------------------------------------------
# 元策略基类（name 保持 'base'，registry 不会收录）
# ----------------------------------------------------------------------

class _EnsembleStrategy(Strategy):
    """ensemble 公共基类：准备基础面板并做线性混合。"""

    universe = "cross_section"
    long_only = True

    def _core_panels(self, data: MarketData) -> List[pd.DataFrame]:
        panels, _ = _snapshot(data)
        return [panels[nm] for nm in core_base_names()]


class _PerfWeightedEnsemble(_EnsembleStrategy):
    """按基础策略历史表现加权的元策略基类（walk-forward，系数滞后一期）。"""

    weighting = "inverse_vol"

    def mix_coefficients(self, data: MarketData) -> pd.DataFrame:
        """返回逐期混合系数面板：index=data.dates，columns=核心基础策略名。

        系数逐行非负且和为 1；第 t 行只用基础策略在 [0, t-1] 的回测收益。
        """
        _, rets = _snapshot(data)
        return _coefficients(rets, self.weighting,
                             window=int(self.params["window"]),
                             min_obs=int(self.params["min_obs"]),
                             vol_floor=float(self.params["vol_floor"]),
                             periods_per_year=int(data.periods_per_year))

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        panels = self._core_panels(data)
        names = core_base_names()
        coef = self.mix_coefficients(data)[names].to_numpy(dtype="float64")
        return _blend(panels, coef, data)


# ----------------------------------------------------------------------
# 元策略
# ----------------------------------------------------------------------

class EqualWeightEnsembleStrategy(_EnsembleStrategy):
    """等权元策略：基础策略权重面板的算术平均。"""

    name = "equal_weight_ensemble"
    channel = "ensemble"
    universe = "cross_section"
    long_only = True
    description = ("等权元策略：把 sma_cross、donchian_turtle、ts_momentum、low_volatility、"
                   "inverse_vol 五个基础策略的目标权重面板按 1/5 等权平均成单一元策略权重"
                   "（各腿先统一为 long_only，系数恒为 1/5，每行权重和 ≤ 1，无杠杆）。")
    hypothesis = ("核心假设：单个策略的失效时点无法预测，而趋势择时（双均线/通道突破）、时序动量、"
                  "低波动异象、逆波动率配置这几类范式的收益驱动来源不同、相关性较低，等权平均能在"
                  "不做任何『哪个策略接下来有效』判断的前提下分散策略层风险，压低元策略波动与单策略"
                  "回撤，长期获得更接近池内平均、更平滑的净值。失效场景：系统性危机中各基础策略暴露"
                  "趋同（趋势与低波同时受挫、相关性趋向 1），分散化红利消失；等权也放弃了对表现差异"
                  "的自适应，长期会跑输池内最优单策略，且在全部基础策略同向亏损时同样亏损。")
    source = ("元策略（strategy of strategies / fund-of-strategies）经典范式——策略层等权组合；"
              "基础策略全部复用本仓库 technical / momentum / factor / allocation 渠道已稳定的公开 "
              "Strategy 类，等权混合、long_only 统一与去杠杆为本渠道原创实现。")
    params = {"weighting": "equal", "n_bases": len(CORE_BASES), "bases": core_base_names()}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        panels = self._core_panels(data)
        coef = _constant_coefficients(data, len(panels))
        return _blend(panels, coef, data)


class InverseVolEnsembleStrategy(_PerfWeightedEnsemble):
    """逆波动元策略：按基础策略截至 t-1 的滚动回测波动倒数加权（walk-forward）。"""

    name = "inverse_vol_ensemble"
    channel = "ensemble"
    universe = "cross_section"
    long_only = True
    weighting = "inverse_vol"
    description = ("逆波动元策略（walk-forward）：先用 engine.Backtester 对 sma_cross、donchian_turtle、"
                   "ts_momentum、low_volatility、inverse_vol 各做一次零成本向量化回测得到策略收益序列，"
                   "第 t 期按各基础策略「截至 t-1」的 window 期滚动波动倒数（1/σ，σ 设下限 vol_floor）"
                   "加权，逐行归一到和为 1 后混合其权重面板；观测不足 min_obs 的预热期回退等权。")
    hypothesis = ("核心假设：策略层波动率的可预测性远强于策略层收益率，把更多资金分给近期更平稳的基础"
                  "策略，能在不预测收益的前提下压低元策略波动、改善夏普（风险平价的策略层版本）；波动"
                  "下限避免长期空仓（σ→0）的策略获得爆炸性权重。系数由滚动统计再 shift(1) 得到，第 t 期"
                  "只使用 [0, t-1] 的已实现回测收益，严格防未来。失效场景：波动 regime 切换时历史波动对"
                  "前瞻波动失去代表性（低波策略恰在波动跳升前重仓）、低波腿本身拥挤，或各基础策略波动"
                  "接近时退化为近似等权而无增益；它也不区分波动的方向性，高收益高波动策略会被系统性低配。")
    source = ("元策略范式——策略层逆波动率加权（inverse-volatility weighting over strategies），"
              "思路来自风险平价/波动率目标；基础策略收益由本仓库 engine.Backtester 回测产生，"
              "滚动窗口、shift(1) 的 walk-forward 系数与归一/退化回退为本渠道原创实现。")
    params = {"weighting": "inverse_vol", "window": 126, "min_obs": 42,
              "vol_floor": 1e-4, "cost_rate": 0.0, "n_bases": len(CORE_BASES),
              "bases": core_base_names()}


class SharpeWeightedEnsembleStrategy(_PerfWeightedEnsemble):
    """夏普加权元策略：按基础策略截至 t-1 的滚动夏普(clip≥0)加权（walk-forward）。"""

    name = "sharpe_weighted_ensemble"
    channel = "ensemble"
    universe = "cross_section"
    long_only = True
    weighting = "sharpe"
    description = ("夏普加权元策略（walk-forward）：对 sma_cross、donchian_turtle、ts_momentum、"
                   "low_volatility、inverse_vol 各做零成本回测得到策略收益序列，第 t 期按各基础策略"
                   "「截至 t-1」的 window 期滚动年化夏普（负值 clip 到 0）作为混合系数，逐行归一到和为 1 "
                   "后加权其权重面板；预热期或全部夏普非正时回退等权。")
    hypothesis = ("核心假设：策略的风险调整后表现存在持续性（manager/strategy momentum）——近期滚动"
                  "夏普高的基础策略，其信号与当前市场性格更契合，把资金倾斜给它可提升元策略的期望夏普；"
                  "clip≥0 保证系数非负（只做多策略、不做空失效策略），归一保证无杠杆。系数由滚动统计再 "
                  "shift(1) 得到，第 t 期只用 [0, t-1] 的回测收益，严格防未来。失效场景：策略收益均值回复"
                  "（近期赢家随即变输家，倾斜反而追高杀低）、窗口过短使夏普估计噪声主导、危机中所有基础"
                  "策略夏普同时为负而被迫等权（此时它退化为 equal_weight_ensemble，无法降低总暴露）。")
    source = ("元策略范式——策略层夏普加权 / 表现倾斜配置（performance-tilted fund-of-strategies），"
              "思路来自 manager momentum 与均值-方差配置的对角近似；滚动夏普、clip≥0、shift(1) 的 "
              "walk-forward 系数与退化回退为本渠道原创实现，基础策略复用本仓库已稳定渠道的公开类。")
    params = {"weighting": "sharpe", "window": 126, "min_obs": 63,
              "vol_floor": 1e-4, "clip_lower": 0.0, "cost_rate": 0.0,
              "n_bases": len(CORE_BASES), "bases": core_base_names()}


class TrendReversionBlendStrategy(_EnsembleStrategy):
    """趋势/均值回归固定比例混合元策略（60/40 风格杠铃）。"""

    name = "trend_reversion_blend"
    channel = "ensemble"
    universe = "cross_section"
    long_only = True
    description = ("趋势-回归混合元策略：趋势腿取 technical 渠道 donchian_turtle（唐奇安通道突破），"
                   "均值回归腿取 meanrev 渠道 zscore_reversion（滚动 z-score 回归，因其可多空，混合时只"
                   "保留多头腿以维持 long_only），按固定 60%/40% 比例逐期线性叠加两者的目标权重面板，"
                   "每行权重和 ≤ 0.6 + 0.4 = 1，无杠杆。")
    hypothesis = ("核心假设：趋势型策略在单边行情盈利、在震荡市被反复止损；均值回归策略恰好相反——震荡市"
                  "收割偏离、单边市逆势受损。两者收益在时间上互补（低相关甚至负相关），按固定 60/40 混合可"
                  "在不预测市场 regime 的前提下平滑净值、降低最大回撤，同时以趋势腿为主保留上行捕获能力。"
                  "失效场景：高波动无方向的跳空行情中两腿同时受损（突破即反转、偏离不回归）；固定比例不随 "
                  "regime 自适应，若市场长期以某一风格为主，混合会持续跑输该风格的纯策略。")
    source = ("元策略范式——趋势跟随与均值回归的风格混合（style blend / barbell of strategies）；"
              "两条腿分别复用本仓库 technical.DonchianTurtleStrategy 与 meanrev.ZscoreReversionStrategy "
              "公开类，多头腿截取、固定比例混合与去杠杆为本渠道原创实现。")
    params = {"weighting": "fixed_blend", "trend_base": "donchian_turtle",
              "reversion_base": "zscore_reversion", "trend_weight": 0.6,
              "reversion_weight": 0.4}

    def blend_coefficients(self) -> Tuple[str, str, float, float]:
        """返回 (趋势腿名, 回归腿名, 趋势权重, 回归权重)，供测试与报告核对。"""
        trend = str(self.params["trend_base"])
        rev = str(self.params["reversion_base"])
        wt = float(self.params["trend_weight"])
        wr = float(self.params["reversion_weight"])
        return trend, rev, wt, wr

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        trend, rev, wt, wr = self.blend_coefficients()
        panels, _ = _snapshot(data)
        legs = [panels[trend], panels[rev]]
        coef = np.tile(np.array([[wt, wr]], dtype="float64"), (len(data.dates), 1))
        return _blend(legs, coef, data)
