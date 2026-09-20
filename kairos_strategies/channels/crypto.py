"""渠道：crypto —— 加密市场风格的原创策略（网格、定投、24/7 动量、carry 代理）。

收集途径：加密市场实践（grid trading、DCA、24/7 momentum、资金费/现货-远期 carry）。
本仓库数据是通用价格面板，此处把它当作加密现货价格序列使用；
全部策略均为基于价格面板的原创实现——不联网、不接任何真实交易所。

工程要点：
- 网格用状态机维护当前档位、定投用时间阶梯累加，辅助函数全部在本模块内实现；
- 所有信号只用「截至当期」的信息（无未来函数），输出确定性；
- long_only：逐资产仓位 ∈ [0,1]，乘以等预算 1/N，每行权重和 ≤ 1，不加杠杆。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy
from .. import indicators as ind


def _budget(data: MarketData) -> float:
    """逐资产等预算：1/N（组合最大满仓，不加杠杆）。"""
    return 1.0 / max(data.n_assets, 1)


def _to_panel(level: np.ndarray, data: MarketData) -> pd.DataFrame:
    """把 [0,1] 的仓位水平矩阵缩放为等预算权重面板，对齐 data 的 index/columns。"""
    lv = np.nan_to_num(np.asarray(level, dtype=float), nan=0.0)
    w = np.clip(lv, 0.0, 1.0) * _budget(data)
    return pd.DataFrame(w, index=data.dates, columns=data.symbols)


def _grid_position(px: np.ndarray, spacing: float, max_steps: int) -> np.ndarray:
    """单资产网格仓位状态机（本模块自实现，不依赖其它渠道的私有函数）。

    以首个有效价格为参考价，按对数等比网格（间距 spacing）划分档位；
    价格每下穿一档持有 +1（低买），每上穿一档持有 -1（高卖），
    持有档数裁剪在 [0, max_steps]，输出归一化仓位 hold/max_steps ∈ [0,1]。
    只用截至当期的价格，无未来函数。
    """
    step = float(np.log1p(spacing))
    out = np.zeros(len(px), dtype=float)
    ref_log = np.nan          # 参考价（对数），取首个有效价格
    level = 0                 # 当前网格档位（价格下穿为正）
    hold = 0                  # 当前持有份数（状态机的状态）
    for t in range(len(px)):
        p = px[t]
        if not np.isfinite(p) or p <= 0.0:
            out[t] = hold / max_steps
            continue
        log_p = float(np.log(p))
        if not np.isfinite(ref_log):
            ref_log = log_p
        new_level = int(np.floor((ref_log - log_p) / step))
        if new_level > level:                    # 下穿了 (new_level-level) 档 -> 加仓
            hold = min(hold + (new_level - level), max_steps)
        elif new_level < level:                  # 上穿了 (level-new_level) 档 -> 减仓
            hold = max(hold - (level - new_level), 0)
        level = new_level
        out[t] = hold / max_steps
    return out


class GridTradingStrategy(Strategy):
    name = "grid_trading"
    channel = "crypto"
    universe = "timing"
    long_only = True
    description = "网格交易：以首个有效价为参考价、按等比间距划档，价格每下穿一档加一份仓位、每上穿一档减一份，仓位在 [0,1] 内阶梯变化（状态机维护当前档位）。"
    hypothesis = "加密市场高波动、宽震荡，网格化『低买高卖』能在区间内持续积累低位筹码；单边急涨时过早清空仓位、单边深跌时满仓承接是主要失效场景。"
    source = "加密市场实践——网格交易 (grid trading)，本仓库基于价格面板原创实现。"
    params = {"spacing": 0.04, "max_steps": 5}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        px = data.prices.to_numpy(dtype=float)
        spacing = float(self.params["spacing"])
        max_steps = max(int(self.params["max_steps"]), 1)
        if px.shape[1] == 0:
            return pd.DataFrame(index=data.dates, columns=data.symbols)
        level = np.column_stack(
            [_grid_position(px[:, j], spacing, max_steps) for j in range(px.shape[1])]
        )
        return _to_panel(level, data)


class DcaStrategy(Strategy):
    name = "dca"
    channel = "crypto"
    universe = "timing"
    long_only = True
    description = "定投 (DCA)：每隔 period 期等额买入一份，仓位随时间阶梯式累加，至 n_tranches 份封顶（模拟分批买入摊薄成本）。"
    hypothesis = "分批买入摊薄成本、放弃择时，假设长期看多且入场时点不可预测；持续单边下跌中会不断『接飞刀』，是其失效场景。"
    source = "加密市场实践——定投 (Dollar-Cost Averaging)，本仓库原创实现。"
    params = {"period": 20, "n_tranches": 10}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        period = max(int(self.params["period"]), 1)
        n_tranches = max(int(self.params["n_tranches"]), 1)
        t = np.arange(len(data.dates))
        tranches = np.minimum(t // period + 1, n_tranches)     # 已买入份数，随时间阶梯递增
        level = np.repeat((tranches / n_tranches)[:, None], data.n_assets, axis=1)
        return _to_panel(level, data)


class CryptoMomentum247Strategy(Strategy):
    name = "crypto_momentum_247"
    channel = "crypto"
    universe = "timing"
    long_only = True
    description = "24/7 动量：加密市场无休市，用较短窗口（7/30 期）时序动量双确认，做多强势币、空仓弱势币，等预算 1/N。"
    hypothesis = "加密资产 7x24 连续交易、信息消化更快，趋势窗口更短，短周期动量在趋势延续段有效；高噪声震荡期频繁进出会失效。"
    source = "加密市场实践——24/7 短周期时序动量，本仓库基于价格面板原创实现。"
    params = {"fast": 7, "slow": 30}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        mom_fast = ind.momentum(p, int(self.params["fast"]))
        mom_slow = ind.momentum(p, int(self.params["slow"]))
        signal = ((mom_fast > 0) & (mom_slow > 0)).astype(float)   # NaN 比较为 False
        return signal.fillna(0.0) * _budget(data)


class CarryProxyStrategy(Strategy):
    name = "carry_proxy"
    channel = "crypto"
    universe = "timing"
    long_only = True
    description = "carry 代理：以「短期均线 - 长期均线」的相对价差作为持有收益 (carry) 的概念性代理，正 carry（短均线高于长均线）时持有、否则空仓，等预算 1/N。"
    hypothesis = "真实的现货-远期基差/资金费在加密市场代表持有收益；此处用价格面板的短长均线价差做概念性代理——正价差近似『持有有正收益』。注意这是概念性代理、非真实资金费数据；震荡市中价差频繁变号会失效。"
    source = "加密市场实践——资金费率/现货-远期 carry 的概念性代理，本仓库基于价格面板原创实现。"
    params = {"fast": 10, "slow": 30, "buffer": 0.0}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        fast_ma = ind.sma(p, int(self.params["fast"]))
        slow_ma = ind.sma(p, int(self.params["slow"]))
        carry = (fast_ma - slow_ma) / slow_ma
        signal = (carry > float(self.params["buffer"])).astype(float)   # NaN 比较为 False
        return signal.fillna(0.0) * _budget(data)
