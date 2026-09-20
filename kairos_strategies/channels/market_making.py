"""渠道：market_making —— 做市 / 流动性提供策略（日频概念性代理，本仓库 100% 原创实现）。

收集途径：经典做市理论与实践 —— Avellaneda-Stoikov (2008) 库存规避做市、
库存偏斜报价 (inventory-skew quoting)、短期流动性提供（为流动性冲击当对手盘）、
网格做市。全部策略为基于日频价格/成交量面板的原创 numpy/pandas 实现，
不联网、不下真实订单、不使用任何第三方策略代码。

**重要：日频概念性代理的近似假设（务必知悉）**
本仓库数据是日频收盘价 + 成交量，没有真实盘口（无买卖双边报价、无挂单簿、
无逐笔成交）。因此本渠道的「做市」是对经典做市模型（Avellaneda-Stoikov 等）的
**日频概念性代理**，并非真实盘口撮合，存在固有模型误差：
1. 中间价 / 参考价（公允价值锚）：用尾部窗口的滚动 VWAP（成交量有效时按量
   加权，否则退化为 SMA）近似；
2. reservation price：按 A-S 框架用「库存 × gamma × 日波动率」调整参考价
   （多库存压低 reservation price、激励卖出），原模型的 σ²(T−t) 项在日频
   适配中以日波动率 σ 替代；
3. 净库存：代替做市商实际持仓，按时间顺序递推的状态变量（买入库存 +1、
   卖出 −1，或以速度 eta 向目标收敛），有界于 [-cap, +cap]；
4. 成交近似：用收盘价相对「报价带 / 网格档位」的位置代替 bid/ask 被击中
   （穿过阈值即视为一次成交）；
5. 输出：目标权重 = (净库存 / 库存上限) × (1/N)，逐资产等预算 1/N，保证
   每行绝对值之和 ≤ 1。库存型策略（avellaneda_stoikov_proxy /
   inventory_skew_mm / grid_mm_daily）的**行和可以不为 0**——净库存对应有界
   的方向性敞口（|行和| ≤ 1）；liquidity_provision_ls 为美元中性截面策略，
   逐行去均值后行和恒 ≈ 0。

防未来函数：参考价 / 波动率均为尾部窗口（min_periods=window，预热期 NaN 时
策略一律保持库存不动、不下注），库存状态机严格按时间顺序递推，q_t 只依赖
q_{t-1} 与 ≤ t 的价格/成交量；截断样本重算得到的前缀权重与全样本逐位一致
（见 tests/test_market_making.py 的防未来测试）。全程离线、无随机、确定性。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy

_EPS = 1e-9          # 价格下限保护（log / 比值运算的安全前提）
_TINY = 1e-12        # 除零 / 退化保护阈值
_SIGMA_FLOOR = 1e-4  # 日波动率下限：恒定价格（零波动）时防标准化偏离爆炸


# ---------------------------------------------------------------------------
# 模块内自实现的辅助函数（不 import 其它渠道的私有工具）
# ---------------------------------------------------------------------------

def _safe_prices(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64 并裁到正数，对齐 data 的 index/columns。"""
    return data.prices.astype("float64").clip(lower=_EPS)


def _get_volumes(data: MarketData) -> Optional[pd.DataFrame]:
    """取与价格对齐的成交量面板；缺失或全 NaN 时返回 None（触发 SMA 回退）。"""
    v = data.volumes
    if v is None:
        return None
    v = v.reindex(index=data.prices.index, columns=data.prices.columns)
    v = v.apply(pd.to_numeric, errors="coerce")
    if not bool(v.notna().any().any()):
        return None
    return v.fillna(0.0).clip(lower=0.0)


def _reference_price(data: MarketData, window: int) -> pd.DataFrame:
    """参考价（日频「中间价」代理）：尾部窗口滚动 VWAP，成交量缺失/退化时回退 SMA。

    VWAP = Σ(price×vol)/Σvol，窗口含当期（min_periods=window），只看截至当期的
    数据；某窗口成交量退化（Σvol ≈ 0）时**逐元素**退回同窗口的 SMA（回退判定
    也只用尾部信息）；预热期为 NaN，策略端一律视为「保持库存不动」。
    """
    p = _safe_prices(data)
    w = max(int(window), 2)
    sma_ref = ind.sma(p, w)
    v = _get_volumes(data)
    if v is None:
        return sma_ref
    pv = (p * v).rolling(w, min_periods=w).sum()
    vv = v.rolling(w, min_periods=w).sum()
    vwap = pv / vv.where(vv > _TINY)
    return vwap.where(vv > _TINY, sma_ref)           # 成交量退化窗口 -> 尾部 SMA


def _daily_vol(p: pd.DataFrame, window: int) -> pd.DataFrame:
    """日频波动率（尾部窗口滚动标准差，非年化）；预热期为 NaN。"""
    w = max(int(window), 2)
    return p.pct_change().rolling(w, min_periods=w).std()


def _budget(data: MarketData) -> float:
    """逐资产等预算 1/N：每资产库存分数 ∈ [-1,1] 时，保证每行绝对值和 ≤ 1。"""
    return 1.0 / max(int(data.n_assets), 1)


def _inventory_to_weights(q: np.ndarray, cap: float, data: MarketData) -> pd.DataFrame:
    """净库存面板 (T×N) -> 目标权重面板：w = clip(q/cap, -1, 1) × (1/N)。

    每行绝对值之和 ≤ 1；库存型策略行和可以不为 0（净库存 = 有界方向性敞口），
    但 |行和| ≤ 1，元信息中已注明。
    """
    c = max(float(cap), _TINY)
    frac = np.clip(np.nan_to_num(np.asarray(q, dtype="float64") / c), -1.0, 1.0)
    return pd.DataFrame(frac * _budget(data), index=data.dates, columns=data.symbols)


# ---------------------------------------------------------------------------
# 策略 1：Avellaneda-Stoikov 做市日频代理
# ---------------------------------------------------------------------------

class AvellanedaStoikovProxyStrategy(Strategy):
    """Avellaneda-Stoikov 做市模型的日频概念性代理（非真实盘口撮合）。

    经典 A-S (2008)：做市商围绕 reservation price ``r = s − q·γ·σ²·(T−t)``
    挂买卖双边报价，库存 q 使 reservation price 向「减少库存」的方向偏移。
    本策略的日频适配（近似假设见模块 docstring）：
    - 中间价 s → 尾部窗口滚动 VWAP/SMA 参考价 ``ref_t``；
    - ``σ²(T−t)`` 项 → 日波动率 ``σ_t``（尾部窗口估计，带下限保护）；
    - reservation price ``r_t = ref_t·(1 − γ·σ_t·q_{t-1}/cap)``：多库存压低
      reservation price（更想卖），空库存抬高（更想买）；
    - 标准化偏离 ``d_t = (p_t − r_t)/(ref_t·σ_t) = (p_t/ref_t − 1)/σ_t + γ·q_{t-1}/cap``；
    - 目标库存 ``q* = −cap·tanh(d_t)``：价格低于 reservation → 建多库存、
      高于 → 建空库存，偏离越大目标越偏向反方向（tanh 有界饱和）；
    - 库存以速度 ``eta`` 向目标收敛：``q_t = q_{t-1} + eta·(q* − q_{t-1})``，
      裁剪在 [−cap, +cap]；``gamma`` 控制库存规避强度（reservation price 的
      库存偏斜），``eta`` 控制库存回归速度；
    - 输出权重 = ``(q_t/cap)·(1/N)``：每行绝对值和 ≤ 1，行和可以不为 0 但有界。

    状态机按时间顺序递推，q_t 只用 ≤ t 的信息；预热期（ref/σ 无效）保持库存不动。
    """

    name = "avellaneda_stoikov_proxy"
    channel = "market_making"
    universe = "timing"
    long_only = False
    description = ("A-S 做市日频代理：reservation price = 滚动 VWAP/均线参考价按 库存×gamma×日波动 "
                   "调整，收盘价相对 reservation 的标准化偏离经 tanh 决定目标库存（价低建多、价高建空、"
                   "偏离越大越反向），库存以 eta 速度向目标收敛并裁剪在 ±cap，权重=(库存/cap)×(1/N)；"
                   "每行绝对值和 ≤ 1，行和可不为 0 但有界（净库存敞口）。")
    hypothesis = ("核心假设：短周期价格变动含流动性冲击成分，围绕公允锚（参考价）按偏离反向建立有界"
                  "库存、并用库存惩罚项使净敞口均值回归，可在价格向参考价回归时获利（做市价差的日频"
                  "代理）。失效场景：单边趋势/结构性位移中价格长期偏离参考价，库存被顶在上限「接飞刀」；"
                  "日频数据无真实盘口，「成交」只是收盘价穿越的概念性近似，赚不到真实 bid-ask 价差，"
                  "代理误差本身是主要模型风险。")
    source = ("Avellaneda-Stoikov (2008) 库存规避做市模型的日频概念性代理，本仓库原创 numpy 实现："
              "reservation price 用滚动 VWAP/SMA + 库存×gamma×日波动调整，σ²(T−t) 以日波动率替代，"
              "非真实盘口撮合。与 meanrev.zscore_reversion（三态择时、无库存递推）、"
              "microstructure.vwap_reversion（只做多、无库存惩罚）、crypto.grid_trading（固定锚、"
              "long-only）在信号构造与组合形态上均不同。")
    params = {"ref_window": 20,     # 参考价（中间价代理）的尾部窗口
              "sigma_window": 20,   # 日波动率估计的尾部窗口
              "gamma": 1.0,         # 库存规避系数：reservation price 受库存偏斜的强度
              "eta": 0.25,          # 库存向目标收敛的速度 ∈ (0, 1]
              "cap": 5.0}           # 库存上限（权重归一分母，净库存有界于 ±cap）

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _safe_prices(data)
        ref = _reference_price(data, int(self.params["ref_window"])).to_numpy(dtype="float64")
        sig = _daily_vol(p, int(self.params["sigma_window"])).to_numpy(dtype="float64")
        P = p.to_numpy(dtype="float64")
        T, N = P.shape
        gamma = max(float(self.params["gamma"]), 0.0)
        eta = float(np.clip(float(self.params["eta"]), 1e-3, 1.0))
        cap = max(float(self.params["cap"]), _TINY)
        q = np.zeros((T, N), dtype="float64")
        cur = np.zeros(N, dtype="float64")
        for t in range(T):                              # 时间顺序递推，只用 ≤ t 的信息
            for j in range(N):
                r, s, px = ref[t, j], sig[t, j], P[t, j]
                if np.isfinite(r) and r > _EPS and np.isfinite(px) and np.isfinite(s):
                    s = max(float(s), _SIGMA_FLOOR)
                    # reservation price 偏离（标准化）：d = (p − r)/(ref·σ) = (p/ref − 1)/σ + γ·q/cap
                    d = (px / r - 1.0) / s + gamma * cur[j] / cap
                    target = -cap * float(np.tanh(d))   # 偏离越大，目标库存越偏向反方向
                    cur[j] = float(np.clip(cur[j] + eta * (target - cur[j]), -cap, cap))
                q[t, j] = cur[j]                        # 信息无效（预热/退化）时保持库存
        return _inventory_to_weights(q, cap, data)


# ---------------------------------------------------------------------------
# 策略 2：库存偏斜做市
# ---------------------------------------------------------------------------

class InventorySkewMmStrategy(Strategy):
    """库存偏斜做市（inventory-skew quoting 的日频概念性代理，非真实盘口撮合）。

    模拟做市商「双边报价 + 按库存偏斜报价」的行为（近似假设见模块 docstring）：
    - 参考价 ``ref_t``（尾部滚动 VWAP/SMA）代替中间价；
    - 报价带：``bid = ref·(1 − band − shift)``、``ask = ref·(1 + band − shift)``，
      其中偏斜 ``shift = skew·band·(q_{t-1}/cap)``：持多库存时整条报价带下移
      → 收盘价更容易触及 ask（卖出成交）、更难触及 bid（买入成交），持空库存
      时反之，从而把净库存均值回归拉回 0；
    - 成交近似：收盘价 ≤ bid → 买入一份（库存 +1，低于参考价买入加库存）；
      收盘价 ≥ ask → 卖出一份（库存 −1，高于参考价卖出减库存，可转空）；
      介于两者之间 → 无成交，库存保持；净库存有界于 [-cap, +cap] 份；
    - 输出权重 = ``(q/cap)·(1/N)``：每行绝对值和 ≤ 1，行和可不为 0 但有界。

    状态机按时间顺序递推，只用 ≤ t 的参考价与收盘价；预热期（参考价无效）不动。
    """

    name = "inventory_skew_mm"
    channel = "market_making"
    universe = "timing"
    long_only = False
    description = ("库存偏斜做市：维护有界净库存，收盘价低于参考价×(1−band−偏斜) 买入一份加库存、"
                   "高于参考价×(1+band−偏斜) 卖出一份减库存（可转空）；偏斜 = skew·band·(库存/cap)，"
                   "多库存时报价带整体下移使其更易卖出，驱动库存均值回归到 0；权重=(库存/cap)×(1/N)，"
                   "行和可不为 0 但有界。")
    hypothesis = ("核心假设：价格围绕参考价（近期成交中枢）震荡时，「低买高卖」的双边成交能持续积累"
                  "库存利润，库存偏斜报价保证净敞口有界且自动向 0 回归，风险可控。失效场景：单边行情中"
                  "价格持续位于报价带同一侧，库存被顶在上限（下跌接飞刀 / 上涨过早转空）；日频无真实"
                  "盘口，「报价被击中」只是收盘价穿越阈值的概念性近似，赚不到真实价差；band 相对日波动"
                  "过宽时几乎不成交、过窄时高频翻转。")
    source = ("做市实务中的库存偏斜报价范式（inventory-skew quoting，如 Avellaneda-Stoikov 的"
              "报价偏斜思想）的日频概念性代理，本仓库原创状态机实现，非真实盘口撮合。与 "
              "crypto.grid_trading（固定首价锚、long-only [0,max_steps]、无库存偏斜）不同：此处参考价"
              "滚动、库存双向有界，且报价带随库存偏斜以驱动库存归零。")
    params = {"ref_window": 20,     # 参考价（中间价代理）的尾部窗口
              "band": 0.01,         # 报价带半宽（相对参考价的百分比）
              "skew": 1.0,          # 库存偏斜强度 ∈ [0,1]：满库存时报价带整体平移 band·skew
              "cap_steps": 5}       # 净库存上限（份），有界于 [-cap, +cap]

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _safe_prices(data)
        ref = _reference_price(data, int(self.params["ref_window"])).to_numpy(dtype="float64")
        P = p.to_numpy(dtype="float64")
        T, N = P.shape
        band = max(float(self.params["band"]), 1e-6)
        skew = float(np.clip(float(self.params["skew"]), 0.0, 1.0))
        cap = max(int(self.params["cap_steps"]), 1)
        q = np.zeros((T, N), dtype="float64")
        cur = np.zeros(N, dtype=int)
        for t in range(T):                              # 时间顺序递推，只用 ≤ t 的信息
            for j in range(N):
                r, px = ref[t, j], P[t, j]
                if np.isfinite(r) and r > _EPS and np.isfinite(px):
                    shift = skew * band * (cur[j] / cap)    # 多库存 -> 报价带下移（更倾向卖出）
                    bid = r * (1.0 - band - shift)
                    ask = r * (1.0 + band - shift)
                    if px <= bid:
                        cur[j] = min(cur[j] + 1, cap)       # 「买单成交」：低于参考价买入加库存
                    elif px >= ask:
                        cur[j] = max(cur[j] - 1, -cap)      # 「卖单成交」：高于参考价卖出减库存/转空
                q[t, j] = float(cur[j])                     # 预热期无成交，库存保持
        return _inventory_to_weights(q, float(cap), data)


# ---------------------------------------------------------------------------
# 策略 3：流动性提供多空（美元中性）
# ---------------------------------------------------------------------------

class LiquidityProvisionLsStrategy(Strategy):
    """流动性提供多空：为极短期流动性冲击提供对手盘（美元中性截面策略）。

    做市商的核心经济功能是当流动性冲击的对手盘：短期被砸下去的资产买入、
    被短期买盘推高的资产卖出，赚取冲击回补的收敛。日频适配：
    - 冲击度量：极短 horizon（1~3 期）收益的「每期平均」
      ``shock = mean_h((p_t/p_{t-h} − 1)/h)``（全部尾部信息）；
    - 冲击强度：按资产自身尾部日波动标准化 ``x = −shock/σ``（反向：短期急跌
      → x 大正 → 做多；急涨 → x 大负 → 做空），截断在 ±z_cap 防单一资产
      吃满预算；σ 退化（恒定价格）或预热期按 0 处理；
    - 组合构造：逐行去截面均值 → 美元中性（行和恒 ≈ 0），再按每行绝对值和
      ≤ budget 整体缩放（不加杠杆）；单资产 universe 无法中性，直接空仓。
    """

    name = "liquidity_provision_ls"
    channel = "market_making"
    universe = "cross_section"
    long_only = False
    description = ("流动性提供多空：对 1~3 期极短收益取每期平均并按自身日波动标准化得到冲击强度，"
                   "做多短期急跌（流动性冲击的买方对手盘）、做空短期急涨（卖方对手盘），逐行去均值"
                   "构成美元中性组合（行和 ≈ 0），每行绝对值和 ≤ 1。")
    hypothesis = ("核心假设：极短窗口（1~3 期）的价格变动中流动性冲击（被动抛压/抢购、止损盘）占比"
                  "最高，冲击造成的临时错价随后回补，站在冲击对面提供流动性可获得近似做市价差的"
                  "补偿；美元中性对冲掉市场方向，收益来源是截面冲击离散度。失效场景：短期涨跌由真实"
                  "信息驱动（业绩爆雷/利好兑现）时跌者继续跌、涨者继续涨，「对手盘」变成接飞刀；"
                  "horizon 极短、换手高，交易成本可能吞掉毛收益；危机期流动性螺旋中冲击持续同向放大。")
    source = ("做市/流动性提供理论（为订单流冲击提供对手盘、赚取流动性溢价）的日频截面代理，本仓库"
              "原创实现。与 long_short.reversal_ls（lookback 5、top/bottom 分位篮子等权）、"
              "statarb.xs_zscore_reversion（lookback 10、截面 z + 平滑）、factor.short_term_reversal"
              "（只做多）不同：此处 horizon 仅 1~3 期、按时序波动标准化成「冲击强度（几个日 σ）」、"
              "连续配权 + 截断 + 逐行去均值，语义是流动性提供而非纯反转因子。")
    params = {"horizons": (1, 2, 3),   # 极短冲击观察期（期数）
              "sigma_window": 20,      # 冲击强度标准化用的尾部日波动窗口
              "z_cap": 3.0,            # 冲击强度截断（防单一资产吃满预算）
              "budget": 1.0}           # 组合毛敞口上限（每行绝对值和）

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        zero = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if int(data.n_assets) < 2:
            return zero                                   # 单资产无法构成美元中性
        p = _safe_prices(data)
        hs: List[int] = [max(int(h), 1) for h in self.params["horizons"]] or [1]
        shock: Optional[pd.DataFrame] = None
        for h in hs:
            per = (p / p.shift(h) - 1.0) / float(h)       # h 期收益的每期平均（尾部）
            shock = per if shock is None else shock + per
        assert shock is not None
        shock = shock / float(len(hs))                    # 极短 horizon 平均每期收益
        sigma = _daily_vol(p, int(self.params["sigma_window"]))
        x = (-shock) / sigma.where(sigma > _TINY)         # 反向 = 当冲击的对手盘；σ 退化 -> NaN
        xv = x.to_numpy(dtype="float64")
        xv[~np.isfinite(xv)] = 0.0                        # 预热 / 退化 -> 无冲击，按 0 参与
        zc = max(float(self.params["z_cap"]), _TINY)
        xv = np.clip(xv, -zc, zc)
        xv = xv - xv.mean(axis=1, keepdims=True)          # 逐行去均值 -> 行和恒 ≈ 0（美元中性）
        budget = max(float(self.params["budget"]), _TINY)
        abs_sum = np.abs(xv).sum(axis=1, keepdims=True)
        scale = np.where(abs_sum > budget, budget / np.maximum(abs_sum, _TINY), 1.0)
        out = xv * scale                                  # 每行绝对值和 ≤ budget（不加杠杆）
        out[~np.isfinite(out)] = 0.0
        return pd.DataFrame(out, index=data.dates, columns=data.symbols)


# ---------------------------------------------------------------------------
# 策略 4：日频网格做市（双向库存）
# ---------------------------------------------------------------------------

class GridMmDailyStrategy(Strategy):
    """日频网格做市：围绕滚动参考价的双向有界库存网格（概念性代理）。

    - 网格锚：尾部滚动参考价 ``ref_t``（VWAP/SMA，每期随窗口重新居中），
      对数等比网格，间距 ``spacing``；
    - 档位：``L_t = floor((ln ref_t − ln p_t)/ln(1+spacing))``，价格低于参考价
      一档 L 加一（高于则为负）；
    - 库存递推：价格每下穿一档 → 净库存 +1 份（买入），每上穿一档 → −1 份
      （卖出，可转空），``q_t = clip(q_{t-1} + (L_t − L_{t-1}), −cap, +cap)``；
      参考价漂移时网格随之重新居中，净库存随时间向「围绕参考价中性」回归；
    - 输出权重 = ``(q/cap)·(1/N)``：每行绝对值和 ≤ 1，行和可不为 0 但有界。

    与 crypto.grid_trading 的区别：后者锚定**首个有效价格**且 long-only
    （仓位 ∈ [0, max_steps]）；本策略锚定**滚动参考价**、库存双向可为空
    （∈ [-cap, +cap]），语义是围绕公允价中性的网格做市而非低位吸筹。
    状态机按时间顺序递推，只用 ≤ t 的价格与参考价。
    """

    name = "grid_mm_daily"
    channel = "market_making"
    universe = "timing"
    long_only = False
    description = ("日频网格做市：围绕滚动参考价（VWAP/均线，每期重新居中）设对数等比网格，价格每"
                   "下穿一档净库存 +1 份、每上穿一档 −1 份（可转空），库存有界于 [-cap, +cap]；"
                   "权重=(库存/cap)×(1/N)，行和可不为 0 但有界。与 crypto.grid_trading（固定首价锚、"
                   "long-only）不同：此处双向、围绕参考价中性。")
    hypothesis = ("核心假设：价格围绕滚动的近期成交中枢震荡，网格化「低买高卖」在震荡区间内持续"
                  "积累库存利润，滚动锚使库存不会因价格中枢漂移而无限累积、净敞口有界且向中性回归。"
                  "失效场景：单边趋势中价格连续穿越同侧网格，库存被顶在上限（深跌满仓承接 / 急涨"
                  "过早转空）；日频无真实盘口，「穿档成交」只是收盘价穿越的概念性近似；spacing 相对"
                  "日波动过宽时成交稀疏、过窄时高频翻转放大换手成本。")
    source = ("网格做市（grid market making）实务的日频概念性代理，本仓库原创状态机实现，非真实"
              "盘口撮合。与 crypto.grid_trading 的区别：参考价滚动（非锚定首价）、库存双向可空"
              "（非 long-only [0,max_steps]）、围绕参考价中性；与 inventory_skew_mm 的区别：库存由"
              "「网格档位穿越数」递推而非报价带触发，且无报价偏斜项。")
    params = {"ref_window": 20,     # 滚动参考价（网格锚）的尾部窗口
              "spacing": 0.02,      # 网格间距（对数等比，2%）
              "cap_steps": 4}       # 净库存上限（份），有界于 [-cap, +cap]

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _safe_prices(data)
        ref = _reference_price(data, int(self.params["ref_window"])).to_numpy(dtype="float64")
        P = p.to_numpy(dtype="float64")
        T, N = P.shape
        step = float(np.log1p(max(float(self.params["spacing"]), 1e-6)))
        cap = max(int(self.params["cap_steps"]), 1)
        q = np.zeros((T, N), dtype="float64")
        cur = np.zeros(N, dtype="float64")     # 当前净库存（份）
        prev = np.zeros(N, dtype="float64")    # 上一期网格档位（价格低于参考价的档数）
        for t in range(T):                     # 时间顺序递推，只用 ≤ t 的信息
            for j in range(N):
                r, px = ref[t, j], P[t, j]
                if np.isfinite(r) and r > _EPS and np.isfinite(px) and px > _EPS:
                    lvl = float(np.floor((np.log(r) - np.log(px)) / step))
                    cur[j] = float(np.clip(cur[j] + (lvl - prev[j]), -cap, cap))
                    prev[j] = lvl              # 下穿一档库存 +1，上穿一档 −1（可转空）
                q[t, j] = cur[j]               # 预热期（参考价无效）库存保持
        return _inventory_to_weights(q, float(cap), data)
