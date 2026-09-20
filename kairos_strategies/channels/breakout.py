"""渠道：breakout —— 突破/通道类择时策略（逐资产 timing，只做多）。

收集途径：经典突破范式的四个**不同变体**（与已有 bollinger_breakout /
donchian_turtle / atr_breakout 明确区分，避免重名与重复逻辑）：
  1. range_breakout      : 震荡区间突破 + 「波动收缩后突破」过滤（squeeze）；
  2. keltner_breakout    : 肯特纳通道（EMA ± k×ATR），上轨进、中轨（带缓冲）出；
  3. volatility_breakout : Larry Williams 波动率突破（前收 + k×近期波幅触发价）+ 时间止损；
  4. channel_atr_breakout: 唐奇安式通道突破，但要求突破幅度 > k×ATR 才确认（过滤假突破）。

数据约束与工程要点：
1. 本仓库只有**收盘价面板**（无 high/low/open）。令 high ≈ low ≈ close，则真实波幅
   TR = max(|H-L|, |H-C_prev|, |L-C_prev|) 退化为 |close_t - close_{t-1}|（「绝对价格
   变动」，等价于 |收益|×价格 的一阶近似），其 Wilder 平滑即收盘价近似 ATR；
   「近期波幅」同理用 |close.diff()| 的简单均值代理。两者都是标准 TR 的下界近似，
   会系统性低估含日内振幅的波动，但用于自适应通道宽度/确认门槛足够。
2. 无开盘价时，波动率突破的「开盘 + k×波幅」触发价改用「前收 + k×波幅」，
   用当日收盘是否站上触发价判定（收盘确认范式）。
3. 防未来函数：所有通道/触发价/ATR/波幅/分位阈值均基于**截至 t-1 的历史**
   （滚动统计后再 shift(1)），第 t 期权重只比较 close_t 与昨日已知的阈值；
   引擎还会再滞后一期，双重保险。
4. 状态机持有 + 进/出双阈值（滞后带）显著降低日频 whipsaw 换手；
   权重恒 >= 0，等预算 1/n_assets，逐行和 <= 1（不加杠杆），`_finalize` 兜底裁剪。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy


# --------------------------------------------------------------------------
# 模块内小工具（自实现，不从其它渠道 import 私有函数）
# --------------------------------------------------------------------------
def _close_atr(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """收盘价近似 ATR：TR ≈ |close.diff()|，再做 Wilder 平滑（ewm alpha=1/window）。

    只有收盘价时 high=low=close，标准 TR 退化为「绝对价格变动」（见模块 docstring）。
    对 DataFrame 逐列独立计算，保持 index/columns 语义；min_periods=window，不足为 NaN。
    """
    tr = close.diff().abs()
    return tr.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()


def _mean_abs_move(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """「近期波幅」代理：|close.diff()| 的 window 日简单均值（不足为 NaN）。"""
    return close.diff().abs().rolling(window, min_periods=window).mean()


def _state_machine(enter: pd.DataFrame, exit_: pd.DataFrame,
                   max_hold: Optional[int] = None) -> pd.DataFrame:
    """状态机持有：enter 置 1 并保持，exit_ 清 0，其间沿用上一状态（滞后带效果）。

    可选 max_hold（时间止损）：连续 max_hold 期未出现**新的 enter 信号**则强制离场，
    用于近似日内突破策略的「次日/限期离场」；若持有期内 enter 再次触发，视为新一次
    突破并重置计时（重新入场与首次入场同义）。exit_ 优先级最高。
    """
    e = enter.fillna(False).values.astype(bool)
    x = exit_.reindex(index=enter.index, columns=enter.columns).fillna(False).values.astype(bool)
    out = np.zeros(e.shape, dtype=float)
    hold = np.zeros(e.shape[1], dtype=float)
    age = np.zeros(e.shape[1], dtype=int)
    for t in range(e.shape[0]):
        row_e, row_x = e[t], x[t]
        for j in range(e.shape[1]):
            if row_x[j]:
                hold[j] = 0.0
                age[j] = 0
            elif row_e[j]:
                hold[j] = 1.0
                age[j] = 1
            elif hold[j] > 0.0:
                age[j] += 1
                if max_hold is not None and age[j] > int(max_hold):
                    hold[j] = 0.0
                    age[j] = 0
        out[t] = hold
    return pd.DataFrame(out, index=enter.index, columns=enter.columns)


def _equal_budget(state: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """0/1 持有状态 -> 等预算权重（每资产 1/N，组合最大满仓，行和 <= 1）。"""
    return state.fillna(0.0) / max(int(data.n_assets), 1)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、去 NaN、裁到 [0, cap]，并把行和超过 cap 的行按比例缩回。"""
    out = w.reindex(index=data.dates, columns=data.symbols)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=cap)
    total = out.sum(axis=1)
    factor = np.where(total > cap, cap / total.replace(0.0, np.nan), 1.0)
    return out.mul(pd.Series(factor, index=out.index), axis=0)


# --------------------------------------------------------------------------
# 策略 1：区间突破 + 波动收缩过滤
# --------------------------------------------------------------------------
class RangeBreakout(Strategy):
    """震荡区间突破：收盘站上过去 N 日区间上沿（滚动 max，滞后一期）做多，跌回区间中轴离场。

    与 technical.donchian_turtle（纯新高进/新低出）的区别：
    离场参考线是**区间中轴**（进/出双阈值滞后带），且可选「区间收窄后才承认突破」的
    波动收缩（squeeze）过滤——只有当前区间宽度不高于其历史分位阈值时突破才有效。
    """

    name = "range_breakout"
    channel = "breakout"
    universe = "timing"
    long_only = True
    description = (
        "区间突破 + 波动收缩过滤：收盘价突破过去 N 日震荡区间上沿（滚动最高价，用截至昨日的"
        "历史）做多，跌回区间中轴（(上沿+下沿)/2）离场；默认要求突破前区间相对宽度不高于其"
        "历史滚动分位（squeeze），即「横盘收窄后的突破」才入场，状态机持有降低换手。"
    )
    hypothesis = (
        "核心假设：长时间窄幅震荡代表供需平衡与筹码沉淀，波动收缩后的方向性突破更可能是"
        "信息驱动的真突破并延续为趋势（波动收缩形态 VCP / 挤压突破的思路）；离场用区间中轴"
        "而非下沿，在趋势衰竭初期即兑现，牺牲少量空间换取更低回撤。"
        "失效场景：无趋势的宽幅震荡市中区间上沿频繁被噪声刺穿（squeeze 过滤可部分缓解）；"
        "收缩过滤会漏掉波动扩张期的突破，V 型反转中反应滞后。"
    )
    source = "经典突破范式——价格区间（箱体）突破 + 波动收缩过滤（本仓库原创实现）。"
    params = {
        "window": 20,            # 震荡区间回看天数 N（滚动最高/最低收盘价）
        "use_squeeze": True,     # 是否启用「区间收窄后突破」过滤
        "squeeze_window": 60,    # 区间宽度分位阈值的统计窗口
        "squeeze_q": 0.5,        # 当前区间宽度 <= 历史该分位 才视为收缩（允许突破）
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        n = int(p["window"])
        # 区间上/下沿与中轴：只用截至昨日的 N 日历史（shift(1) 防未来函数）
        upper = close.rolling(n, min_periods=n).max().shift(1)
        lower = close.rolling(n, min_periods=n).min().shift(1)
        mid = (upper + lower) / 2.0
        width = (upper - lower) / mid.where(mid > 0.0)      # 相对区间宽度（量纲无关）
        valid = upper.notna() & lower.notna() & mid.notna()
        enter = (close > upper) & valid                     # 收盘站上区间上沿
        if bool(p["use_squeeze"]):
            sw = int(p["squeeze_window"])
            hist = width.shift(1)                           # 宽度阈值只用昨日之前的历史
            thr = hist.rolling(sw, min_periods=max(2, sw // 2)).quantile(float(p["squeeze_q"]))
            enter = enter & (width <= thr) & thr.notna()    # 收缩（含宽度恒 0 的极端横盘）才放行
        exit_ = (close < mid) & valid                       # 跌回区间中轴离场（滞后带）
        return _finalize(_equal_budget(_state_machine(enter, exit_), data), data)


# --------------------------------------------------------------------------
# 策略 2：肯特纳通道突破
# --------------------------------------------------------------------------
class KeltnerBreakout(Strategy):
    """肯特纳通道突破：EMA ± k×ATR（收盘价近似 ATR），收盘突破上轨做多、回落中轨下方离场。

    与 volatility.atr_breakout（SMA 中轨、无入场缓冲）的区别：中轨用 **EMA**（肯特纳
    通道的定义特征，对近期价格更敏感），且进/出各加一个比例缓冲带（滞后带）：
    入场需 close > 上轨×(1+entry_band)，离场需 close < 中轨×(1-exit_band)。
    """

    name = "keltner_breakout"
    channel = "breakout"
    universe = "timing"
    long_only = True
    description = (
        "肯特纳通道突破：中轨 = EMA(span)，上/下轨 = EMA ± k×ATR（只有收盘价，"
        "ATR 用 |close.diff()| 的 Wilder 平滑近似，即平均绝对价格变动，见模块 docstring）；"
        "收盘突破上轨（带 entry_band 缓冲）做多，回落至中轨下方（带 exit_band 缓冲）离场，"
        "状态机持有。"
    )
    hypothesis = (
        "核心假设：EMA 中轨 + 波动自适应带宽能刻画「相对近期均衡的异常强势」——收盘站上"
        "上轨说明涨速超出近期波动常态，趋势惯性使其大概率延续；跌回 EMA 中轨意味着异常"
        "强势消失，及时离场。进/出双缓冲带避免价格在轨道附近抖动导致的频繁换手。"
        "失效场景：无方向的高波动震荡市（通道被反复刺穿）；收盘近似 ATR 系统性偏窄，"
        "触发比真实高低价通道更频繁；单边急跌后的 V 型反弹会因 EMA 滞后而踏空。"
    )
    source = "经典技术分析——肯特纳通道 (Keltner Channel) 突破范式（本仓库原创实现）。"
    params = {
        "ema_span": 20,          # 中轨 EMA 跨度
        "atr_window": 14,        # ATR（Wilder）窗口
        "k": 2.0,                # 带宽倍数：上轨 = EMA + k×ATR
        "entry_band": 0.0,       # 入场缓冲：close > 上轨×(1+entry_band) 才触发
        "exit_band": 0.005,      # 离场缓冲：close < 中轨×(1-exit_band) 才离场（滞后带）
        "atr_floor_pct": 0.002,  # ATR 下限 = 0.2%×前收，避免零波动时通道退化为中轨本身
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        mid = ind.ema(close, int(p["ema_span"]))
        atr = _close_atr(close, int(p["atr_window"]))
        # ATR 下限用前一日收盘（防未来函数）：完全平坦的价格序列 |diff|=0 会使通道失效
        atr = np.maximum(atr, float(p["atr_floor_pct"]) * close.shift(1))
        k = float(p["k"])
        upper_prev = (mid + k * atr).shift(1)               # 用昨日通道判断今日收盘
        mid_prev = mid.shift(1)
        valid = upper_prev.notna() & mid_prev.notna()
        enter = (close > upper_prev * (1.0 + float(p["entry_band"]))) & valid
        exit_ = (close < mid_prev * (1.0 - float(p["exit_band"]))) & valid
        return _finalize(_equal_budget(_state_machine(enter, exit_), data), data)


# --------------------------------------------------------------------------
# 策略 3：波动率突破（Larry Williams 范式，收盘价适配 + 时间止损）
# --------------------------------------------------------------------------
class VolatilityBreakout(Strategy):
    """波动率突破：触发价 = 前收 + k×近期波幅，收盘站上触发价做多，回落大阴线或限期离场。

    经典 Larry Williams 波动率突破用「当日开盘 + k×昨日波幅」的盘中停止单；本仓库
    无开盘价，改用「**前一日收盘** + k×近期波幅」作触发价、以当日收盘是否站上触发价
    确认（收盘确认范式）。「近期波幅」用 |close.diff()| 的 M 日均值代理（见模块 docstring）。
    离场 = 「回落大阴线」（收盘跌破 前收 - k_exit×波幅）或时间止损 max_hold 期，
    后者近似日内突破「次日离场」的短持范式（max_hold=2 即约等于隔日离场）。
    """

    name = "volatility_breakout"
    channel = "breakout"
    universe = "timing"
    long_only = True
    description = (
        "波动率突破（Larry Williams 范式的收盘价适配）：触发价 = 前收 + k_entry×近期波幅"
        "（|close.diff()| 的 M 日均值，滞后一期），当日收盘站上触发价做多；收盘跌破"
        "「前收 - k_exit×波幅」的回落大阴线立即离场，否则最多持有 max_hold 期强制离场"
        "（时间止损，近似日内策略的次日/限期离场），持有期内触发价再次突破则重置计时。"
    )
    hypothesis = (
        "核心假设：单日超出近期常态波幅的向上位移是「日内动能爆发」的信号，动量在短期内"
        "（数日）延续，但会快速衰减——因此用时间止损限期持有、见反向大阴线立即离场，"
        "把每次突破当作一笔短线的动能交易而非趋势跟踪。"
        "失效场景：突破后立即反转的假突破/猎杀止损行情（靠 k_exit 反向触发价快速止血）；"
        "低波动背景下的微小位移也会触发（用 range_floor_pct 波幅下限缓解）；"
        "持续单边趋势中因限期离场而跑输趋势跟踪策略。"
    )
    source = "经典日内突破范式——Larry Williams 波动率突破 (volatility breakout) 的收盘价适配（本仓库原创实现）。"
    params = {
        "range_window": 10,       # 「近期波幅」统计窗口 M（|close.diff()| 简单均值）
        "k_entry": 1.2,           # 入场触发倍数：前收 + k_entry×波幅
        "k_exit": 1.2,            # 回落离场倍数：前收 - k_exit×波幅
        "max_hold": 10,           # 时间止损：无新突破信号连续持有超过 max_hold 期强制离场
        "range_floor_pct": 0.002, # 波幅下限 = 0.2%×前收，防止零波幅使触发价退化为前收
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        prev = close.shift(1)                               # 前一日收盘
        rng_ = _mean_abs_move(close, int(p["range_window"]))
        rng_ = np.maximum(rng_, float(p["range_floor_pct"]) * prev)
        rng_prev = rng_.shift(1)                            # 波幅只用截至昨日的历史
        trig_up = prev + float(p["k_entry"]) * rng_prev     # 入场触发价（昨日已知）
        trig_dn = prev - float(p["k_exit"]) * rng_prev      # 回落离场价（昨日已知）
        valid = trig_up.notna() & trig_dn.notna()
        enter = (close > trig_up) & valid
        exit_ = (close < trig_dn) & valid
        state = _state_machine(enter, exit_, max_hold=int(p["max_hold"]))
        return _finalize(_equal_budget(state, data), data)


# --------------------------------------------------------------------------
# 策略 4：通道突破 + ATR 幅度确认（过滤假突破）
# --------------------------------------------------------------------------
class ChannelAtrBreakout(Strategy):
    """唐奇安式通道突破 + ATR 幅度确认：突破幅度必须 > k×ATR 才入场（过滤假突破）。

    与 technical.donchian_turtle 的区别：入场不是「收盘 > 上轨」即成立，而是要求
    **超出量** (close - 上轨) > k×ATR（昨日值）——仅被噪声刺穿通道上沿的假突破
    因幅度不足被拒绝；离场用独立的 M 日低点通道（进/出双通道滞后带）。
    """

    name = "channel_atr_breakout"
    channel = "breakout"
    universe = "timing"
    long_only = True
    description = (
        "通道 + ATR 过滤突破：收盘价突破过去 N 日最高收盘（唐奇安式上轨，滞后一期）且"
        "**突破幅度 > k×ATR**（收盘价近似 ATR，滞后一期）才确认做多；跌破过去 M 日最低"
        "收盘（退出通道）离场。幅度确认拒绝仅被噪声刺穿上沿的假突破，状态机持有。"
    )
    hypothesis = (
        "核心假设：真突破应伴随「超出常态波幅」的位移——以 ATR 为尺子给突破幅度设门槛，"
        "能过滤大部分仅由噪声造成的通道刺穿（假突破的主要形态），代价是入场价更差、"
        "漏掉缓慢阴涨型的真突破。进（N 日高点 + 幅度确认）出（M 日低点）双通道构成"
        "宽滞后带，显著降低换手。失效场景：波动骤升期门槛随 ATR 抬高导致踏空；"
        "长期缓慢上行但从不单日大幅突破的资产会被系统性忽略。"
    )
    source = "经典突破范式——唐奇安通道突破 + 波动自适应幅度确认（本仓库原创实现）。"
    params = {
        "entry_window": 20,       # 入场通道：过去 N 日最高收盘（滞后一期）
        "exit_window": 10,        # 退出通道：过去 M 日最低收盘（滞后一期）
        "atr_window": 14,         # ATR（Wilder）窗口
        "k": 0.5,                 # 确认倍数：close - 上轨 > k×ATR 才入场
        "atr_floor_pct": 0.002,   # ATR 下限 = 0.2%×前收，避免零波动时门槛退化
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        n, m = int(p["entry_window"]), int(p["exit_window"])
        upper = close.rolling(n, min_periods=n).max().shift(1)   # 昨日已知的通道上沿
        lower = close.rolling(m, min_periods=m).min().shift(1)   # 昨日已知的退出通道
        atr = _close_atr(close, int(p["atr_window"]))
        atr = np.maximum(atr, float(p["atr_floor_pct"]) * close.shift(1))
        margin = float(p["k"]) * atr.shift(1)                    # 确认门槛（昨日已知）
        valid = upper.notna() & lower.notna() & margin.notna()
        enter = ((close - upper) > margin) & valid               # 幅度确认的突破
        exit_ = (close < lower) & valid                          # 跌破退出通道离场
        return _finalize(_equal_budget(_state_machine(enter, exit_), data), data)
