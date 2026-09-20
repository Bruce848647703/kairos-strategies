"""渠道：taa —— 组合层「战术资产配置」(Tactical Asset Allocation) 策略。

与既有渠道的分工：
  - ``allocation`` 解决「钱在风险资产之间怎么分」（每行和 ≈ 1，恒满仓）；
  - ``momentum`` / ``volatility`` 多为单资产择时或敞口 overlay；
  - 本渠道解决**组合层的两个战术问题**：① 风险资产之间怎么切换（截面相对强弱）、
    ② 风险资产整体与**现金**之间怎么切换（总敞口 0~1）。

「现金」的表示方式（本渠道统一约定）：
  权重面板**每行之和 ≤ 1**，``1 - Σw`` 即现金比例；现金不占列、收益记 0。
  因此所有策略都是 long_only（权重 ≥ 0）、不加杠杆，风险资产内部用「固定槽位预算」
  （每个槽位 1/n_slots）而非「归一到 1」——被绝对动量/趋势过滤掉的槽位直接留现金，
  总敞口随之下降。这与 ``momentum.dual_momentum``（选中者归一到满仓）在语义上根本不同。

四个策略（均为本仓库原创实现，仅用 numpy/pandas）：
  1. ``dual_momentum_taa``  双动量 TAA：相对动量选截面最强的 n_slots 个资产，
     再用「绝对动量 vs 现金收益」过滤，不合格的槽位配现金。
  2. ``trend_regime_taa``   趋势状态 TAA：等权「市场指数」在长期均线之上 → 满仓风险
     sleeve（等权或动量加权），之下 → 全部转现金（risk-off），带缓冲滞后带减少抖动。
  3. ``vol_target_taa``     组合波动率目标 TAA：对逆波动率 sleeve 按
     目标波动 / 已实现 sleeve 波动（EWMA）缩放总敞口，上限 1，其余为现金。
  4. ``mom_12m_taa``        经典 12-1 动量 TAA：过去约 252 期（跳过近 21 期）动量选
     top-N 等权，每 21 期再平衡，其间买入持有漂移（按净值归一，天然不产生杠杆），
     动量为负的槽位用现金替代。

防未来函数：第 t 期权重只用截至 t（含 t 收盘）的价格/收益——滚动均线窗口为
``[t-window+1, t]``，动量为 ``p[t-skip]/p[t-skip-lookback]-1``，波动率为历史滚动/EWMA；
预热期（统计量未知）一律**不建仓**（全现金），绝不用未来数据回填。引擎还会再滞后一期。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-12


# ----------------------------------------------------------------------
# 模块内自实现的小工具（不从其它渠道 import 私有函数）
# ----------------------------------------------------------------------

def _prices(data: MarketData) -> pd.DataFrame:
    """价格面板（float64）。"""
    return data.prices.astype("float64")


def _returns(data: MarketData) -> pd.DataFrame:
    """单期收益率面板（NaN → 0）。"""
    return data.returns(1)


def _sma(x: pd.Series, window: int) -> pd.Series:
    """简单移动平均：窗口 ``[t-window+1, t]``，样本不足则 NaN（不回填、不偷看未来）。"""
    w = max(int(window), 1)
    return x.rolling(w, min_periods=w).mean()


def _market_index(data: MarketData) -> pd.Series:
    """「市场」代理：等权每日再平衡组合的净值指数（起点 1.0，确定性）。"""
    r = _returns(data).mean(axis=1)
    return (1.0 + r).cumprod()


def _trailing_mom(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """区间动量：``p[t-skip] / p[t-skip-lookback] - 1``，只用截至 t-skip 的历史价。"""
    lb, sk = max(int(lookback), 1), max(int(skip), 0)
    return prices.shift(sk) / prices.shift(sk + lb) - 1.0


def _cash_return(lookback: int, cash_annual: float, periods_per_year: int) -> float:
    """现金在 lookback 期内的累计收益（年化现金利率按期换算），作为绝对动量门槛。"""
    years = max(int(lookback), 1) / float(max(int(periods_per_year), 1))
    return (1.0 + float(cash_annual)) ** years - 1.0


def _n_slots(n_assets: int, top_frac: float) -> int:
    """槽位数 = ceil(N × top_frac)，至少 1、至多 N（槽位预算 1/n_slots，总敞口 ≤ 1）。"""
    n = max(int(n_assets), 1)
    k = int(np.ceil(n * float(top_frac) - 1e-9))
    return int(min(max(k, 1), n))


def _top_n_mask(factor: pd.DataFrame, n_slots: int) -> pd.DataFrame:
    """每行选出 factor 最大的前 n_slots 个资产（布尔掩码）。

    并列用 ``method='first'`` 按列顺序确定性打破；NaN（预热期）一律不选。
    """
    valid = factor.notna()
    n_valid = valid.sum(axis=1).to_numpy(dtype=int)
    k = np.minimum(int(n_slots), n_valid)                       # 有效资产不足时取全部
    rank = factor.rank(axis=1, ascending=False, method="first").to_numpy()
    mask = (rank <= k[:, None]) & valid.to_numpy()
    return pd.DataFrame(mask, index=factor.index, columns=factor.columns)


def _slot_budget(selected: pd.DataFrame, n_slots: int) -> pd.DataFrame:
    """固定槽位预算：每个选中槽位 1/n_slots，落选槽位保持 0（= 配现金）。

    行和 = 选中槽位数 / n_slots ≤ 1，这正是「风险资产 vs 现金」的战术开关。
    """
    return selected.astype(float) / float(max(int(n_slots), 1))


def _equal_sleeve(data: MarketData) -> pd.DataFrame:
    """等权风险 sleeve：每资产 1/N，行和 = 1（总敞口由上层缩放决定）。"""
    n = max(int(data.n_assets), 1)
    return pd.DataFrame(np.full((len(data.dates), n), 1.0 / n),
                        index=data.dates, columns=data.symbols)


def _momentum_sleeve(data: MarketData, window: int) -> pd.DataFrame:
    """动量加权风险 sleeve：权重 ∝ 截断为正的滚动动量，行内归一到 1。

    全市场无正动量（或动量未知）时回退等权——sleeve 只决定「风险资产之间怎么分」，
    是否持有风险资产由上层的 regime/敞口开关决定，二者职责分离。
    """
    mom = _trailing_mom(_prices(data), window)
    pos = mom.where(mom > 0.0, 0.0).fillna(0.0)
    total = pos.sum(axis=1)
    vals = pos.div(total.where(total > _EPS), axis=0).fillna(0.0).to_numpy()
    n = max(int(data.n_assets), 1)
    vals = np.where(total.to_numpy()[:, None] > _EPS, vals, 1.0 / n)
    return pd.DataFrame(vals, index=data.dates, columns=data.symbols)


def _inverse_vol_sleeve(data: MarketData, window: int, vol_floor: float) -> pd.DataFrame:
    """逆波动率风险 sleeve：权重 ∝ 1/滚动年化波动，行内归一到 1。

    波动未知（预热）的资产暂不进 sleeve；全部未知时回退等权。
    """
    vol = (_prices(data).pct_change()
           .rolling(window, min_periods=max(int(window), 2)).std()
           * np.sqrt(float(data.periods_per_year)))
    floor = max(float(vol_floor), _EPS)
    inv = (1.0 / vol.clip(lower=floor)).where(vol.notna(), 0.0).fillna(0.0)
    total = inv.sum(axis=1)
    vals = inv.div(total.where(total > _EPS), axis=0).fillna(0.0).to_numpy()
    n = max(int(data.n_assets), 1)
    vals = np.where(total.to_numpy()[:, None] > _EPS, vals, 1.0 / n)
    return pd.DataFrame(vals, index=data.dates, columns=data.symbols)


def _ewma_vol(ret: pd.Series, span: int, periods_per_year: int) -> pd.Series:
    """EWMA 已实现波动（年化）：sqrt(EWMA(r²) × ppy)，样本不足则 NaN。"""
    sp = max(int(span), 2)
    var = ret.pow(2).ewm(span=sp, adjust=False, min_periods=sp).mean()
    return np.sqrt(var.clip(lower=0.0) * float(periods_per_year))


def _regime_switch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """组合层 0/1 状态机：enter 置 1 并保持，exit 清 0（离场优先，风险为先）。

    进出用不同阈值即形成滞后带，避免在均线附近反复抖动。
    """
    e = enter.fillna(False).to_numpy(dtype=bool)
    x = exit_.reindex(index=enter.index).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(e), dtype=float)
    hold = 0.0
    for t in range(len(e)):
        if x[t]:
            hold = 0.0
        elif e[t]:
            hold = 1.0
        out[t] = hold
    return pd.Series(out, index=enter.index)


def _drift_hold(prev: np.ndarray, ret: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """买入持有漂移一期，并按组合净值归一（现金收益记 0）。

    ``w' = w ⊙ (1+r) / (1 + w·r)``：分母是组合净值因子，因此 Σw' = (S + r_p)/(1 + r_p)，
    当 S = Σw ≤ 1 时必有 Σw' ≤ 1 —— 漂移**天然不产生杠杆**，现金比例随行情自然变化。
    净值因子非正（极端行情）时返回全 0（视为已止损到现金）。
    """
    grown = np.clip(prev * (1.0 + ret), 0.0, None)
    nav = float(np.dot(prev, ret)) + 1.0
    if not np.isfinite(nav) or nav <= _EPS:
        return np.zeros_like(grown)
    w = grown / nav
    total = float(w.sum())
    return w * (cap / total) if total > cap else w


def _rebalance_drift(target: pd.DataFrame, data: MarketData, rebal: int,
                     cap: float = 1.0) -> pd.DataFrame:
    """定期再平衡 + 其间买入持有漂移的装配（用于低频 TAA，降低换手）。"""
    k = max(int(rebal), 1)
    tgt = target.to_numpy(dtype=float)
    rets = np.nan_to_num(
        _returns(data).reindex(index=target.index, columns=target.columns)
        .to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n, N = tgt.shape
    out = np.zeros((n, N), dtype=float)
    w = np.zeros(N, dtype=float)
    for t in range(n):
        if t % k == 0:
            w = np.clip(tgt[t], 0.0, None)                  # 再平衡日：切到最新目标权重
        else:
            w = _drift_hold(w, rets[t], cap)                # 非再平衡日：买入持有漂移
        out[t] = w
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、NaN → 0、裁到 [0, cap]，并把行和超 cap 的行等比缩回。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0).clip(lower=0.0, upper=cap))
    total = out.sum(axis=1)
    scale = (cap / total.where(total > _EPS)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(scale, axis=0)


# ----------------------------------------------------------------------
# 策略 1：双动量 TAA（相对动量选强 + 绝对动量对现金过滤）
# ----------------------------------------------------------------------

class DualMomentumTaaStrategy(Strategy):
    """双动量 TAA：截面相对动量选出最强槽位，绝对动量不及现金的槽位改配现金。"""

    name = "dual_momentum_taa"
    channel = "taa"
    universe = "cross_section"
    long_only = True
    description = ("双动量战术配置：相对动量在截面选出最强的 n_slots 个资产，每个槽位固定预算 "
                   "1/n_slots；再用绝对动量（自身 lookback 期收益 > 同期现金累计收益）过滤，不合格"
                   "的槽位直接配现金（权重 0），因此总敞口 = 合格槽位数/n_slots ≤ 1。")
    hypothesis = ("核心假设：资产间存在相对强弱的延续性（横截面动量），同时市场整体存在方向性"
                  "（绝对动量/时序动量）——先选最强、再要求它跑赢现金，可在普跌时自动把资金"
                  "撤到现金而不是『矬子里拔将军』，显著压低尾部回撤。以固定槽位预算代替归一化，"
                  "使现金比例成为连续的风险旋钮。失效场景：动量崩溃（急跌后 V 型反转，刚刚"
                  "转现金即踏空）、宽幅震荡市中强弱频繁轮换带来的换手成本、以及现金利率接近 0 时"
                  "绝对动量门槛过低导致过滤形同虚设。")
    source = ("TAA 经典范式——Antonacci 双动量 (dual momentum / GEM) 的**组合层**改写："
              "把『股/债/现金三选一』推广为『截面 top-N 槽位 + 现金门槛过滤』，本仓库原创实现。")
    params = {
        "lookback": 126,          # 动量回看期（约半年）
        "top_frac": 1.0 / 3.0,    # 相对动量槽位占比（N=8 → 3 个槽位）
        "cash_annual": 0.02,      # 现金年化收益（绝对动量门槛：需跑赢现金）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        lookback = int(p["lookback"])
        n_slots = _n_slots(data.n_assets, p["top_frac"])
        mom = _trailing_mom(_prices(data), lookback)
        rel_ok = _top_n_mask(mom, n_slots)                        # 相对动量：截面最强
        hurdle = _cash_return(lookback, p["cash_annual"], data.periods_per_year)
        abs_ok = (mom > hurdle) & mom.notna()                     # 绝对动量：跑赢现金
        return _finalize(_slot_budget(rel_ok & abs_ok, n_slots), data)


# ----------------------------------------------------------------------
# 策略 2：趋势状态 TAA（市场指数 vs 长期均线 → 风险资产 / 现金）
# ----------------------------------------------------------------------

class TrendRegimeTaaStrategy(Strategy):
    """趋势状态 TAA：等权市场指数在长期均线上方满仓风险 sleeve，下方全部转现金。"""

    name = "trend_regime_taa"
    channel = "taa"
    universe = "cross_section"
    long_only = True
    description = ("趋势状态战术配置：以等权组合净值指数作为『市场』代理，指数高于其 ma_window 期"
                   "均线（含 ±buffer 滞后带，状态机持有）时满仓持有风险 sleeve（等权或正动量加权），"
                   "跌破均线时全部转为现金（risk-off，总敞口 0）。")
    hypothesis = ("核心假设：大类资产/宽基市场存在可持续数月的趋势状态（牛熊切换），长期均线是"
                  "最简单稳健的状态判别器——在均线之上时承担风险、之下时退到现金，可截断熊市"
                  "左尾、以少量震荡市的假信号成本换取回撤的大幅下降。buffer 滞后带 + 状态机持有"
                  "降低在均线附近反复进出的换手。失效场景：无方向的箱体震荡（连续假突破、反复"
                  "挨打）、V 型急跌急涨（均线滞后导致既没躲过下跌又踏空反弹）、以及均线窗口与"
                  "市场周期错配时。")
    source = ("TAA 经典范式——趋势跟随 / 200 日均线择时 (trend regime、time-series momentum "
              "overlay) 的多资产组合层实现，本仓库原创。")
    params = {
        "ma_window": 200,     # 长期均线窗口（约 10 个月）
        "buffer": 0.01,       # 滞后带：上穿 ma×(1+buffer) 进场，下破 ma×(1-buffer) 离场
        "sleeve": "equal",    # 风险 sleeve 内部配置："equal" 等权 / "momentum" 正动量加权
        "mom_window": 63,     # sleeve="momentum" 时的动量窗口（约一季度）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        ma_window = max(int(p["ma_window"]), 2)
        buffer = abs(float(p["buffer"]))
        mkt = _market_index(data)
        ma = _sma(mkt, ma_window)
        ready = ma.notna()                                  # 预热期均线未知 → 不建仓
        enter = ready & (mkt > ma * (1.0 + buffer))
        exit_ = ready & (mkt < ma * (1.0 - buffer))
        risk_on = _regime_switch(enter, exit_)              # 0/1 组合层开关
        if str(p["sleeve"]) == "momentum":
            sleeve = _momentum_sleeve(data, int(p["mom_window"]))
        else:
            sleeve = _equal_sleeve(data)
        w = sleeve.mul(risk_on.reindex(data.dates).fillna(0.0), axis=0)
        return _finalize(w, data)


# ----------------------------------------------------------------------
# 策略 3：组合波动率目标 TAA（逆波动率 sleeve × 目标波动缩放 → 余额为现金）
# ----------------------------------------------------------------------

class VolTargetTaaStrategy(Strategy):
    """波动率目标 TAA：按 目标波动/已实现 sleeve 波动 缩放总敞口，余额留现金。"""

    name = "vol_target_taa"
    channel = "taa"
    universe = "cross_section"
    long_only = True
    description = ("组合波动率目标战术配置：风险 sleeve 用滚动逆波动率加权（行内归一到 1），"
                   "再按 scale = clip(目标波动 / sleeve 已实现波动(EWMA 年化), 0, 1) 缩放总敞口，"
                   "scale < 1 的差额即为现金；波动越高仓位越低，缩放上限 1（不加杠杆）。")
    hypothesis = ("核心假设：波动率有强聚集性与可预测性，且高波动期单位风险补偿更差（去杠杆、"
                  "流动性收缩、杠杆效应），把组合波动锁定在目标水平能显著改善风险调整后收益并"
                  "压低下行尾部。与单资产择时不同，这里直接对**组合 sleeve 的已实现波动**（含相关"
                  "性效应）做目标控制，比用单资产波动近似更准确；EWMA 对波动突变反应快于等权滚动"
                  "窗口。失效场景：目标波动长期高于市场已实现波动时退化为满仓持有（不加杠杆故无"
                  "超额收益）、波动骤升后急速回落的 V 型行情中因滞后而踏空、以及波动与收益正相关"
                  "的上行加速阶段会被系统性降仓。")
    source = ("TAA 风险管理范式——组合层 volatility targeting / 风险预算缩放（目标波动 - 现金"
              "二元配置），EWMA 波动与逆波动率 sleeve 均为本仓库原创实现。")
    params = {
        "target_vol": 0.10,      # 年化目标组合波动
        "vol_span": 20,          # EWMA 波动半衰期参数（span）
        "sleeve_window": 60,     # 逆波动率 sleeve 的波动估计窗口
        "vol_floor": 0.02,       # 年化波动下限，防止近零波动把敞口顶爆
        "max_scale": 1.0,        # 敞口上限：1 = 不加杠杆，差额为现金
        "min_scale": 0.0,        # 敞口下限：0 = 允许完全转现金
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        sleeve = _inverse_vol_sleeve(data, int(p["sleeve_window"]), float(p["vol_floor"]))
        # sleeve 已实现收益用「上一期 sleeve 权重 × 本期资产收益」，与引擎时序一致
        held = sleeve.shift(1).fillna(0.0)
        sleeve_ret = (held * _returns(data)).sum(axis=1)
        rv = _ewma_vol(sleeve_ret, int(p["vol_span"]), data.periods_per_year)
        rv = rv.clip(lower=max(float(p["vol_floor"]), _EPS))
        scale = (float(p["target_vol"]) / rv).clip(lower=float(p["min_scale"]),
                                                   upper=float(p["max_scale"]))
        scale = scale.reindex(data.dates).fillna(0.0)        # 波动未知 → 不建仓（现金）
        return _finalize(sleeve.mul(scale, axis=0), data)


# ----------------------------------------------------------------------
# 策略 4：经典 12-1 动量 TAA（低频再平衡 + 现金替代）
# ----------------------------------------------------------------------

class Mom12mTaaStrategy(Strategy):
    """12-1 动量 TAA：252 期（跳过近 21 期）动量选 top-N，每 21 期再平衡，负动量配现金。"""

    name = "mom_12m_taa"
    channel = "taa"
    universe = "cross_section"
    long_only = True
    description = ("经典 12-1 动量战术配置：按 p[t-21]/p[t-273]-1（约 12 个月动量、剔除近 1 个月"
                   "反转）截面排序，做多前 n_slots 个资产、每槽固定预算 1/n_slots；每 rebal 期再平衡"
                   "一次，其间买入持有按净值漂移（不产生杠杆）；动量 ≤ 0 的槽位用现金替代，"
                   "故总敞口 = 合格槽位数/n_slots ≤ 1。")
    hypothesis = ("核心假设：12 个月减 1 个月的动量是横截面动量最经典的窗口——长窗口捕捉持续性"
                  "资金流与基本面渐进扩散，跳过最近 1 个月规避短期反转与流动性冲击的污染。低频"
                  "（月度）再平衡 + 期间漂移把换手与交易成本压到最低，符合真实 TAA 组合的运作方式；"
                  "叠加『负动量转现金』的绝对门槛后，熊市里组合自动降敞口而非被动持有最抗跌者。"
                  "失效场景：动量崩溃（政策底后的急速普涨反转）、市场风格季度内切换、以及资产数"
                  "很少时 top-N 的截面区分度不足。")
    source = ("TAA 经典范式——Jegadeesh-Titman 12-1 横截面动量的**月度再平衡 + 现金替代**组合层"
              "实现（含买入持有漂移的净值归一递推），本仓库原创。")
    params = {
        "lookback": 252,          # 动量窗口（约 12 个月）
        "skip": 21,               # 跳过近端（约 1 个月），规避短期反转
        "top_frac": 0.25,         # 持有槽位占比（N=8 → 2 个槽位）
        "rebal": 21,              # 再平衡周期（约每月）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        lookback, skip = int(p["lookback"]), int(p["skip"])
        n_slots = _n_slots(data.n_assets, p["top_frac"])
        mom = _trailing_mom(_prices(data), lookback, skip)
        rel_ok = _top_n_mask(mom, n_slots)
        abs_ok = (mom > 0.0) & mom.notna()                  # 负动量槽位 → 现金
        target = _slot_budget(rel_ok & abs_ok, n_slots)
        w = _rebalance_drift(target, data, int(p["rebal"]))
        return _finalize(w, data)


# 便于外部（报告/研究记录）按渠道枚举的辅助常量
CHANNEL = "taa"
STRATEGY_NAMES = (
    "dual_momentum_taa",
    "trend_regime_taa",
    "vol_target_taa",
    "mom_12m_taa",
)

__all__ = [
    "DualMomentumTaaStrategy",
    "TrendRegimeTaaStrategy",
    "VolTargetTaaStrategy",
    "Mom12mTaaStrategy",
    "CHANNEL",
    "STRATEGY_NAMES",
]
