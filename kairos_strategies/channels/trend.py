"""渠道：trend —— CTA 趋势跟随策略（逐资产 timing，只做多）。

收集途径：经典 CTA / 管理期货 (managed futures) 的趋势跟随范式——
时序动量 + 波动率目标、均线带排列、趋向强度 (ADX) 过滤、Dual Thrust 区间突破、
海龟唐奇安突破 + ATR 仓位。全部为本仓库 100% 原创实现，仅用 numpy/pandas，离线且确定性。

数据约束与工程要点：
1. 本仓库只提供**收盘价面板**（无 high/low/open），因此经典指标做收盘价近似：
   - ATR：令 high≈low≈close，真实波幅 TR=|close_t − close_{t-1}|（绝对价格变动），
     ATR 为其 Wilder(ewm alpha=1/window) 平滑——这是标准 TR 的下界近似（无日内振幅）。
   - Dual Thrust：HH/LC/HC/LL 均退化为收盘价的滚动最高/最低，故 range=max(HH−LC, HC−LL)
     收敛为「滚动收盘极差」；当日 open 用前一日收盘 shift(1) 近似。
   - ADX/DMI：只有收盘价时用价格动能近似 +DI/−DI——
     +DM=max(Δclose,0)、−DM=max(−Δclose,0)，DI=100×Wilder(DM)/ATR，
     DX=100×|+DI−(−DI)|/(+DI+(−DI))，ADX=DX 的 Wilder 平滑（0~100，越大趋势越强）。
2. 防未来函数：所有 MA/ATR/波动/通道/极差均为**滚动历史**统计；突破通道再 shift(1)
   （用昨日通道判断今日收盘）；第 t 期权重只依赖截至 t（含 t 收盘）的信息，引擎还会再滞后一期。
3. 权重恒 ≥ 0；每资产上限=等预算 1/N，故逐行和 ≤ 1（不加杠杆）；`_finalize` 做兜底裁剪与行内缩放。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-12


# --------------------------------------------------------------------------
# 模块内自实现的小工具（不从其它渠道 import 私有函数）
# --------------------------------------------------------------------------
def _close_atr(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """收盘价近似 ATR：TR=|close.diff()|，Wilder(ewm alpha=1/window) 平滑，逐列作用。

    以 high=low=close 代入标准 TR=max(|H−L|,|H−C_prev|,|L−C_prev|) 即退化为 |Δclose|，
    因此结果等价于「平均绝对价格变动」，对 DataFrame 按列独立计算（ewm 天然逐列）。
    """
    tr = close.diff().abs()
    return tr.ewm(alpha=1.0 / float(window), adjust=False, min_periods=int(window)).mean()


def _state_machine(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """状态机持有：enter 置 1 并保持，exit_ 清 0，其间沿用上一状态。

    进入/退出使用不同阈值（滞后带）可显著降低日频 whipsaw 换手。
    """
    e = enter.reindex(index=exit_.index, columns=exit_.columns).fillna(False).values.astype(bool)
    x = exit_.fillna(False).values.astype(bool)
    out = np.zeros(e.shape, dtype=float)
    hold = np.zeros(e.shape[1], dtype=float)
    for t in range(e.shape[0]):
        hold = np.where(x[t], 0.0, np.where(e[t], 1.0, hold))
        out[t] = hold
    return pd.DataFrame(out, index=exit_.index, columns=exit_.columns)


def _equal_budget(state: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """0/1 持有状态 -> 等预算权重（每资产 1/N，组合最大满仓）。"""
    return state.fillna(0.0) / max(int(data.n_assets), 1)


def _finalize(w: pd.DataFrame, data: MarketData, cap: float = 1.0) -> pd.DataFrame:
    """对齐 index/columns、去 NaN、裁到 [0, cap]，并把行和超过 cap 的行按比例缩回（兜底）。"""
    out = w.reindex(index=data.dates, columns=data.symbols)
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=cap)
    total = out.sum(axis=1)
    factor = (cap / total.replace(0.0, np.nan)).where(total > cap, 1.0).fillna(1.0)
    return out.mul(factor, axis=0)


# --------------------------------------------------------------------------
# 策略 1：时序动量 + 波动率缩放
# --------------------------------------------------------------------------
class TsMomVolScaled(Strategy):
    """时序动量 + 波动率目标：过去 N 期收益定方向，仓位按 目标波动/已实现波动 缩放（上限=等预算）。"""

    name = "tsmom_volscaled"
    channel = "trend"
    universe = "timing"
    long_only = True
    description = (
        "时序动量 + 波动率目标：过去 N 期收益为正则做多（long_only），"
        "单资产仓位 = 等预算(1/N) × 趋势强度(mom/mom_ref 截断到[0,1]) × min(目标波动/已实现波动, 1)，"
        "趋势越强、波动越低则仓位越大，单资产上限=1/N，逐行和≤1。"
    )
    hypothesis = (
        "核心假设：资产价格存在中期正自相关（趋势延续），且单位风险收益在低波动期更优。"
        "用过去 N 期收益符号做绝对动量过滤（只持有上涨资产），再用 目标波动/已实现波动 做风险预算，"
        "把每个资产的风险贡献拉平——同样涨幅下低波动更可能源于持续资金流而非噪声，故低波多配。"
        "趋势强度项让强趋势获得更大敞口、弱趋势小仓试探。失效场景：趋势急速反转(V 型)时动量滞后挨打；"
        "无方向宽幅震荡市中反复进出累积成本；目标波动远高于已实现波动时缩放触顶(=1/N)，退化为等预算持有。"
    )
    source = "CTA 时序动量 (time-series momentum) + 波动率目标 (volatility targeting) 范式（本仓库原创实现）。"
    params = {
        "mom_window": 60,      # 动量回看窗口（期）
        "mom_ref": 0.10,       # 趋势强度参考：N 期收益达到该值即视为满强度
        "vol_window": 20,      # 已实现波动窗口
        "target_vol": 0.15,    # 年化目标波动
        "vol_floor": 0.02,     # 已实现波动下限，避免近零波动把敞口顶爆
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        close = data.prices.astype("float64")
        n = max(int(data.n_assets), 1)
        mom = close.pct_change(int(p["mom_window"]))                       # 仅用历史价格
        strength = (mom / float(p["mom_ref"])).clip(lower=0.0, upper=1.0)  # mom<=0 -> 0（long_only）
        strength = strength.where(mom.notna(), 0.0).fillna(0.0)
        rv = ind.realized_vol(close, int(p["vol_window"]), data.periods_per_year)
        rv = rv.clip(lower=float(p["vol_floor"]))
        vscale = (float(p["target_vol"]) / rv).clip(lower=0.0, upper=1.0)  # 波动越低越大，上限 1（不加杠杆）
        vscale = vscale.where(rv.notna(), 0.0).fillna(0.0)
        w = (strength * vscale) / n                                        # 每资产 ≤ 1/N -> 行和 ≤ 1
        return _finalize(w, data)


# --------------------------------------------------------------------------
# 策略 2：均线带（MA Ribbon）
# --------------------------------------------------------------------------
class MaRibbon(Strategy):
    """均线带：多条均线多头排列(短>中>长)时做多，排列分数(正确相邻对占比)越高仓位越大。"""

    name = "ma_ribbon"
    channel = "trend"
    universe = "timing"
    long_only = True
    description = (
        "均线带 (MA Ribbon)：对一组均线(默认 10/20/50/100)统计相邻「短均线在长均线之上」的对数占比作为排列分数∈[0,1]，"
        "完美多头排列分数=1；单资产仓位 = 等预算(1/N) × 排列分数，逐行和≤1。"
    )
    hypothesis = (
        "核心假设：当短、中、长期均线自上而下依次排列(多头排列)时，多周期趋势方向一致，趋势更可信、更可能延续；"
        "用「排列分数」而非单一金叉做连续度量，可在趋势部分对齐时小仓、完全对齐时满仓，平滑信号、降低假突破。"
        "失效场景：均线本质滞后，趋势末端与急速反转时排列尚未翻转已回吐利润；无趋势的横盘缠绕期分数在临界附近抖动。"
    )
    source = "经典技术分析——均线带 / 多周期移动平均排列 (moving-average ribbon)（本仓库原创实现）。"
    params = {
        "windows": [10, 20, 50, 100],   # 由短到长的均线窗口序列
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.prices.astype("float64")
        n = max(int(data.n_assets), 1)
        wins: List[int] = [int(w) for w in self.params["windows"]]
        mas = [ind.sma(close, w) for w in wins]
        score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        valid = pd.DataFrame(True, index=close.index, columns=close.columns)
        cnt = 0
        for i in range(len(mas) - 1):
            short, lng = mas[i], mas[i + 1]
            pair_valid = short.notna() & lng.notna()
            score = score + (short > lng).astype(float)   # 短均线在长均线之上记 1
            valid = valid & pair_valid
            cnt += 1
        score = (score / max(cnt, 1)) * valid.astype(float)   # 排列分数∈[0,1]，均线未就绪则 0
        w = score / n                                         # 每资产 ≤ 1/N -> 行和 ≤ 1
        return _finalize(w, data)


# --------------------------------------------------------------------------
# 策略 3：趋向强度过滤（自研 ADX 近似）
# --------------------------------------------------------------------------
class AdxTrend(Strategy):
    """自研趋向强度 (ADX 近似) 过滤：仅当趋势强度高(ADX>阈值)且方向向上(+DI>−DI)时做多。"""

    name = "adx_trend"
    channel = "trend"
    universe = "timing"
    long_only = True
    description = (
        "用收盘价近似构造 Wilder DMI/ADX：+DM=max(Δclose,0)、−DM=max(−Δclose,0)，TR=|Δclose|，"
        "DI=100×Wilder(DM)/ATR，DX=100×|+DI−(−DI)|/(+DI+(−DI))，ADX=DX 的 Wilder 平滑；"
        "仅当 ADX>阈值(趋势够强)且 +DI>−DI(方向向上)时做多，单资产仓位=等预算 1/N，逐行和≤1。"
    )
    hypothesis = (
        "核心假设：趋势「强度」可与趋势「方向」分离——ADX 度量方向移动相对总波幅的占比，高 ADX 代表单边行情、"
        "低 ADX 代表无方向震荡；只在强趋势且方向向上时持有，可避开震荡市的反复止损。"
        "只有收盘价时用价格动能近似 +DI/−DI：单日收盘涨则全部计入 +DM、跌则计入 −DM，"
        "ATR 退化为平均绝对价格变动，DX=|平滑(+DM)−平滑(−DM)|/(平滑(+DM)+平滑(−DM))×100 仍是有意义的方向占比度量。"
        "失效场景：ADX 高度滞后，趋势启动初期 ADX 仍低而踏空、见顶后才升高而追高；窄幅盘整后突然爆发时反应慢。"
    )
    source = "经典 CTA——Wilder ADX/DMI 趋向系统（本仓库以收盘价近似原创实现）。"
    params = {
        "window": 14,        # DI/DX/ADX 的 Wilder 平滑窗口
        "adx_min": 25.0,     # 趋势强度阈值：ADX 高于此值才认为有趋势
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        w = int(p["window"])
        n = max(int(data.n_assets), 1)
        close = data.prices.astype("float64")
        diff = close.diff()
        up = diff.clip(lower=0.0)              # 收盘价近似 +DM
        dn = (-diff).clip(lower=0.0)           # 收盘价近似 −DM
        tr = diff.abs()                        # 收盘价近似 TR
        alpha = 1.0 / float(w)
        atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        plus_dm = up.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        minus_dm = dn.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        atr_safe = atr_w.where(atr_w > 0.0, np.nan)
        plus_di = 100.0 * plus_dm / atr_safe
        minus_di = 100.0 * minus_dm / atr_safe
        denom = (plus_di + minus_di)
        dx = (100.0 * (plus_di - minus_di).abs() / denom.where(denom > 0.0, np.nan)).fillna(0.0)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        valid = adx.notna() & plus_di.notna() & minus_di.notna()
        hold = (adx > float(p["adx_min"])) & (plus_di > minus_di) & valid
        return _finalize(_equal_budget(hold.astype(float), data), data)


# --------------------------------------------------------------------------
# 策略 4：Dual Thrust 区间突破
# --------------------------------------------------------------------------
class DualThrust(Strategy):
    """Dual Thrust：近 N 日 range×系数构造上下轨，突破上轨做多、跌破下轨离场（状态机持有）。"""

    name = "dual_thrust"
    channel = "trend"
    universe = "timing"
    long_only = True
    description = (
        "Dual Thrust：range = 近 N 日 max(HH−LC, HC−LL)（收盘价近似=滚动收盘极差），"
        "锚定前一日收盘构造 上轨=anchor+K1×range、下轨=anchor−K2×range（range 用截至昨日的值，防未来）；"
        "收盘突破上轨做多、跌破下轨离场，状态机持有，单资产仓位=等预算 1/N，逐行和≤1。"
    )
    hypothesis = (
        "核心假设：以近几日真实波幅自适应地设定突破门槛，可在波动放大时抬高门槛过滤噪声、在波动收敛后更易捕捉真突破；"
        "Dual Thrust 的非对称上下轨(K1/K2)允许对多空设不同敏感度（本策略 long_only 只用做多侧与离场侧）。"
        "只有收盘价时 HH≈HC=滚动最高、LL≈LC=滚动最低，range 退化为滚动收盘极差，当日 open 用前收近似。"
        "失效场景：无方向宽幅震荡市中频繁触发上下轨而反复止损(靠偶发大趋势回本)；range 滞后于波动骤升时门槛偏低产生假突破。"
    )
    source = "经典 CTA——Dual Thrust 区间突破系统（本仓库以收盘价近似原创实现）。"
    params = {
        "window": 4,             # range 回看窗口（日）
        "k1": 0.5,               # 上轨系数
        "k2": 0.5,               # 下轨系数
        "range_floor_pct": 0.002,  # range 下限 = 0.2%×价格，避免完全平坦市 range=0
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        N = int(p["window"])
        n = max(int(data.n_assets), 1)
        close = data.prices.astype("float64")
        hh = close.rolling(N, min_periods=N).max()      # 收盘价近似 HH/HC
        ll = close.rolling(N, min_periods=N).min()      # 收盘价近似 LL/LC
        rng = (hh - ll)                                  # = max(HH−LC, HC−LL) 的收盘价退化形式
        floor = float(p["range_floor_pct"]) * close
        rng = np.maximum(rng, floor)                     # NaN 自动传播（早期窗口不足）
        rng_prev = rng.shift(1)                          # 用截至昨日的 range，防未来
        anchor = close.shift(1)                          # 当日 open 近似 = 前一日收盘
        upper = anchor + float(p["k1"]) * rng_prev
        lower = anchor - float(p["k2"]) * rng_prev
        valid = upper.notna() & lower.notna()
        enter = (close > upper) & valid
        exit_ = (close < lower) & valid
        hold = _state_machine(enter, exit_)
        return _finalize(_equal_budget(hold, data), data)


# --------------------------------------------------------------------------
# 策略 5：海龟 + ATR 仓位
# --------------------------------------------------------------------------
class TurtleAtr(Strategy):
    """海龟唐奇安突破 + ATR 仓位：突破入场、跌破退出通道离场，每资产仓位按 ATR 反比缩放。"""

    name = "turtle_atr"
    channel = "trend"
    universe = "timing"
    long_only = True
    description = (
        "海龟 + ATR 仓位：唐奇安通道突破入场(收盘创 N 日新高做多、跌破 M 日新低离场，通道 shift(1) 防未来，状态机持有)，"
        "每资产仓位 = 等预算(1/N) × 持有(0/1) × min(risk_target/(ATR/价格), 1)，"
        "即按 ATR 占价比反比缩放——波动大仓位小、波动小仓位大，单资产上限=1/N，逐行和≤1。"
    )
    hypothesis = (
        "核心假设：区间突破后趋势倾向延续(海龟交易法则)，而每笔头寸的风险应由波动决定——"
        "用 ATR 占价比作分母做反比缩放(等价于每资产等风险预算)，可在高波动资产上自动减仓、低波动资产上加仓，"
        "平滑组合风险、抑制单一名义敞口主导。入场/离场用不同长度通道(进 N 出 M)形成滞后带降低 whipsaw。"
        "失效场景：震荡市中突破频繁失败造成连续小亏(靠偶发大趋势回本)；ATR 收盘价近似系统性低估含日内振幅的波动；"
        "risk_target 设置过高会让多数资产仓位触顶(=1/N)而丧失波动率区分度。"
    )
    source = "经典 CTA——海龟交易法 (唐奇安突破) + ATR 波动率仓位管理（本仓库原创实现）。"
    params = {
        "entry": 20,            # 入场通道：N 日新高
        "exit": 10,             # 离场通道：M 日新低
        "atr_window": 14,       # ATR（Wilder）窗口
        "risk_target": 0.02,    # 目标风险：仓位 = risk_target / (ATR/价格)，截断到 [0,1]
        "atr_floor_pct": 0.001, # ATR 占价比下限，避免近零波动把仓位顶爆
    }

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = self.params
        n = max(int(data.n_assets), 1)
        close = data.prices.astype("float64")
        eN, xN = int(p["entry"]), int(p["exit"])
        upper = close.rolling(eN, min_periods=eN).max().shift(1)   # 昨日 N 日高点
        lower = close.rolling(xN, min_periods=xN).min().shift(1)   # 昨日 M 日低点
        enter = (close > upper) & upper.notna()
        exit_ = (close < lower) & lower.notna()
        hold = _state_machine(enter, exit_)
        atr = _close_atr(close, int(p["atr_window"]))
        atr_pct = (atr / close).clip(lower=float(p["atr_floor_pct"]))
        size = (float(p["risk_target"]) / atr_pct).clip(lower=0.0, upper=1.0)
        size = size.where(atr_pct.notna(), 0.0).fillna(0.0)        # 波动越大仓位越小，上限 1
        w = (hold * size) / n                                      # 每资产 ≤ 1/N -> 行和 ≤ 1
        return _finalize(w, data)
