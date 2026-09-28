"""多资产配置策略（跨资产类别：A股/海外股/黄金/国债/货币）。

这些策略面向**真实多资产 ETF universe**（见 realdata.ASSET_CLASSES），
故意不放在 channels/ 下，因此不会被 registry 自动纳入「个股」管线，
仅由 examples/run_multi.py 在多资产面板上显式调用。

- AllWeather : 类别层风险平价（逆波动），偏债偏分散，全天候思路的自研简化版。
- GTAA       : 全球战术资产配置——类别层绝对+相对动量选强，弱则转现金(货币ETF)。
- SixtyForty : 经典 60/40（60% 股票 / 40% 债券）定期再平衡基准。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import MarketData, Strategy
from . import realdata


def _present_classes(data: MarketData, asset_classes: Dict[str, List[str]],
                     include: Optional[List[str]] = None,
                     exclude: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """筛出在当前 data.symbols 中实际存在的类别成员。"""
    out = {}
    for c, mem in asset_classes.items():
        if include is not None and c not in include:
            continue
        if exclude is not None and c in exclude:
            continue
        present = [s for s in mem if s in set(data.symbols)]
        if present:
            out[c] = present
    return out


class AllWeather(Strategy):
    name = "all_weather"
    channel = "multi_asset"
    universe = "cross_section"
    long_only = True
    description = "全天候(类风险平价)：在风险资产类别间按逆波动分配，类别内等权。"
    hypothesis = "低相关多资产 + 按风险而非资金配置，可在不同宏观环境下更稳健。"
    source = "多资产配置——All Weather / 风险平价思想（类别层逆波动简化版）。"

    def __init__(self, asset_classes: Optional[Dict[str, List[str]]] = None, window: int = 60):
        self.asset_classes = asset_classes or realdata.ASSET_CLASSES
        self.params = {"window": window}
        self.window = window

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        rets = data.returns()
        classes = _present_classes(data, self.asset_classes, exclude=["cash"])
        w = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if not classes:
            return w
        cr = pd.DataFrame({c: rets[mem].mean(axis=1) for c, mem in classes.items()})
        vol = cr.rolling(self.window, min_periods=self.window).std() * np.sqrt(data.periods_per_year)
        inv = 1.0 / vol.replace(0.0, np.nan)
        cw = inv.div(inv.sum(axis=1), axis=0)
        # 预热期(波动未知)回退类别等权
        eq = 1.0 / len(classes)
        cw = cw.fillna(eq)
        for c, mem in classes.items():
            for s in mem:
                w[s] = cw[c] / len(mem)
        return w.clip(lower=0.0)


class GTAA(Strategy):
    name = "gtaa"
    channel = "multi_asset"
    universe = "cross_section"
    long_only = True
    description = "全球战术资产配置：类别层 12 月绝对+相对动量选强，弱者转现金(货币ETF)。"
    hypothesis = "资产类别存在中期动量；绝对动量过滤可在熊市 retreat 到现金降回撤。"
    source = "多资产配置——GTAA / 双动量(Antonacci)思想。"

    def __init__(self, asset_classes: Optional[Dict[str, List[str]]] = None,
                 lookback: int = 252, top_k: int = 2, vol_window: int = 60):
        self.asset_classes = asset_classes or realdata.ASSET_CLASSES
        self.params = {"lookback": lookback, "top_k": top_k, "vol_window": vol_window}
        self.lookback, self.top_k, self.vol_window = lookback, top_k, vol_window

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        rets = data.returns()
        risk_classes = _present_classes(data, self.asset_classes, exclude=["cash"])
        cash_classes = _present_classes(data, self.asset_classes, include=["cash"])
        cash_syms = [s for mem in cash_classes.values() for s in mem]
        w = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        if not risk_classes:
            return w
        idx = (1.0 + pd.DataFrame({c: rets[mem].mean(axis=1) for c, mem in risk_classes.items()})).cumprod()
        mom = idx.pct_change(self.lookback)
        cash_ret = rets[cash_syms].mean(axis=1) if cash_syms else pd.Series(0.0, index=data.dates)
        cash_mom = (1.0 + cash_ret).cumprod().pct_change(self.lookback)
        vol = (pd.DataFrame({c: rets[mem].mean(axis=1) for c, mem in risk_classes.items()})
               .rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(data.periods_per_year))
        names = list(risk_classes.keys())
        M = mom.values; V = vol.values; CM = cash_mom.values
        out = np.zeros((len(data.dates), len(names)))
        for t in range(len(data.dates)):
            row_mom = M[t]; row_vol = V[t]; cm = CM[t]
            if np.all(~np.isfinite(row_mom)):
                continue  # 预热期空仓(全现金，下方分配)
            order = np.argsort([-1e18 if not np.isfinite(row_mom[j]) else row_mom[j] for j in range(len(names))])[::-1]
            picked, used = [], 0.0
            for j in order:
                if used >= self.top_k:
                    break
                if np.isfinite(row_mom[j]) and (not np.isfinite(cm) or row_mom[j] > cm) and row_mom[j] > 0:
                    picked.append(j); used += 1
            if not picked:
                continue  # 全部转现金
            ivs = []
            for j in picked:
                v = row_vol[j]
                ivs.append(1.0 / v if np.isfinite(v) and v > 1e-9 else 0.0)
            tot = sum(ivs)
            for j, iv in zip(picked, ivs):
                out[t, j] = (iv / tot) if tot > 0 else 1.0 / len(picked)
        # 类别权重 -> 资产权重；未分配部分给现金
        for j, c in enumerate(names):
            mem = risk_classes[c]
            for s in mem:
                w[s] = w[s] + pd.Series(out[:, j], index=data.dates) / len(mem)
        risk_used = pd.Series(out.sum(axis=1), index=data.dates)
        cash_w = (1.0 - risk_used).clip(lower=0.0)
        if cash_syms:
            for s in cash_syms:
                w[s] = w[s] + cash_w / len(cash_syms)
        return w.clip(lower=0.0)


class SixtyForty(Strategy):
    name = "sixty_forty"
    channel = "multi_asset"
    universe = "cross_section"
    long_only = True
    description = "经典 60/40：60% 股票(中外宽基等权) + 40% 债券，定期再平衡。"
    hypothesis = "股债负/低相关，60/40 是长期稳健的配置基准。"
    source = "多资产配置——传统 60/40 股债组合。"

    def __init__(self, asset_classes: Optional[Dict[str, List[str]]] = None, equity: float = 0.6):
        self.asset_classes = asset_classes or realdata.ASSET_CLASSES
        self.params = {"equity": equity}
        self.equity = equity

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        eq_classes = _present_classes(data, self.asset_classes, include=["cn_equity", "global_equity"])
        bond_classes = _present_classes(data, self.asset_classes, include=["bond"])
        cash_classes = _present_classes(data, self.asset_classes, include=["cash"])
        eq_syms = [s for mem in eq_classes.values() for s in mem]
        bond_syms = [s for mem in bond_classes.values() for s in mem]
        cash_syms = [s for mem in cash_classes.values() for s in mem]
        w = pd.DataFrame(0.0, index=data.dates, columns=data.symbols)
        eq_w = self.equity
        bond_w = 1.0 - self.equity
        if not bond_syms:  # 无债券则用现金替代
            bond_syms = cash_syms
        if eq_syms:
            for s in eq_syms:
                w[s] = eq_w / len(eq_syms)
        if bond_syms:
            for s in bond_syms:
                w[s] = w[s] + bond_w / len(bond_syms)
        return w.clip(lower=0.0)


MULTI_ASSET_STRATEGIES = [AllWeather, GTAA, SixtyForty]
