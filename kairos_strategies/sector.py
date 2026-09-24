"""行业中性 / 行业均衡权重处理（在策略原始权重之上做行业层后处理）。

两种模式：
- sector_equalize: 保持只做多与每期总敞口不变，把权重在「当期实际持有的行业」间均衡，
  消除行业集中度（不凭空创造未持有行业的位置）。
- sector_neutralize: 行业内去均值，使每个行业净敞口≈0、整体美元中性（多空），
  再把每期绝对权重和缩放到 ≤1。

均为纯函数、确定性、逐期(按行)处理，不引入未来信息。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _sector_of(col: str, sector_map: Dict[str, str]) -> str:
    return sector_map.get(col, "other")


def sector_equalize(weights: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """在当期实际持有的行业之间均衡权重，保持总敞口与只做多属性。"""
    w = weights.astype("float64").clip(lower=0.0)
    sectors = pd.Series({c: _sector_of(c, sector_map) for c in w.columns})
    out = w.copy()
    vals = w.values
    res = np.zeros_like(vals)
    sec_arr = sectors.values
    for t in range(vals.shape[0]):
        row = vals[t]
        total = row.sum()
        if total <= 0:
            continue
        # 各行业当期毛敞口
        gross: Dict[str, float] = {}
        for j, s in enumerate(sec_arr):
            gross[s] = gross.get(s, 0.0) + row[j]
        active = [s for s, g in gross.items() if g > 1e-12]
        if not active:
            continue
        target = total / len(active)
        scale = {s: (target / gross[s]) for s in active}
        for j, s in enumerate(sec_arr):
            res[t, j] = row[j] * scale.get(s, 0.0)
    out[:] = res
    return out


def sector_neutralize(weights: pd.DataFrame, sector_map: Dict[str, str],
                      cap: float = 1.0) -> pd.DataFrame:
    """行业内去均值 -> 行业&美元中性；再把每期绝对权重和缩放到 ≤cap。"""
    w = weights.astype("float64").fillna(0.0)
    sectors = pd.Series({c: _sector_of(c, sector_map) for c in w.columns})
    # 按行业分组去均值（仅对该行业当期有持仓的列）
    demean = w.copy()
    for s in sectors.unique():
        cols = list(sectors[sectors == s].index)
        sub = w[cols]
        active = sub.abs().sum(axis=1) > 1e-12
        mean = sub.mean(axis=1)
        demean.loc[active, cols] = sub[active].sub(mean[active], axis=0)
        demean.loc[~active, cols] = 0.0
    # 缩放每期绝对和到 ≤cap
    gross = demean.abs().sum(axis=1).replace(0.0, np.nan)
    scale = (cap / gross).clip(upper=1.0).fillna(0.0)
    return demean.mul(scale, axis=0)


def make_sector_map(sector_groups: Dict[str, list]) -> Dict[str, str]:
    """把 {sector: [symbols]} 翻转为 {symbol: sector}。"""
    return {sym: sec for sec, members in sector_groups.items() for sym in members}
