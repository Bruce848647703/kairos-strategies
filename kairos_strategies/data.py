"""市场数据构造：可复现的合成 universe（默认）+ CSV 加载（可选真实数据）。

合成数据刻意植入不同「市场性格」（趋势 / 均值回归 / 随机），
使不同类别策略能表现出差异，便于演示与测试。全程离线、固定 seed 可复现。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import MarketData

REGIMES = ["trend", "meanrev", "random"]


def make_synthetic_universe(n_assets: int = 8,
                            n_days: int = 1000,
                            seed: int = 2026,
                            start: str = "2018-01-02",
                            periods_per_year: int = 252) -> MarketData:
    """生成多资产合成行情（价格 + 成交量）。

    每个资产按索引轮换分配一种性格：
      trend   : AR(1) 正自相关收益 -> 利于动量/趋势类
      meanrev : OU 过程（对数价格向均值回归）-> 利于均值回归类
      random  : 普通 GBM -> 作为对照
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    dt = 1.0 / periods_per_year
    prices = {}
    volumes = {}
    for i in range(n_assets):
        sym = f"A{i}"
        regime = REGIMES[i % len(REGIMES)]
        sigma = 0.18 + 0.10 * ((i * 37) % 5) / 4.0   # 0.18~0.28 年化波动，确定性
        s0 = 50.0 + 10.0 * (i % 4)
        if regime == "trend":
            phi = 0.22                              # 正自相关（趋势性格）
            mu = 0.12 + 0.05 * (i % 3)
            r = np.zeros(n_days)
            eps = rng.standard_normal(n_days) * sigma * np.sqrt(dt)
            for t in range(1, n_days):
                r[t] = mu * dt + phi * r[t - 1] + eps[t]
            logp = np.log(s0) + np.cumsum(r)
        elif regime == "meanrev":
            theta = 2.5                             # 回归速度（均值回归性格）
            mu_log = np.log(s0)
            logp = np.zeros(n_days)
            logp[0] = mu_log
            eps = rng.standard_normal(n_days) * sigma * np.sqrt(dt)
            for t in range(1, n_days):
                logp[t] = logp[t - 1] + theta * (mu_log - logp[t - 1]) * dt + eps[t]
        else:
            mu = 0.05
            eps = rng.standard_normal(n_days) * sigma * np.sqrt(dt)
            logp = np.log(s0) + np.cumsum(mu * dt + eps)
        prices[sym] = np.exp(logp)
        base_vol = 1e6 * (1 + 0.5 * ((i * 13) % 3))
        volumes[sym] = base_vol * np.exp(0.3 * rng.standard_normal(n_days)) + 1e5
    p = pd.DataFrame(prices, index=idx)
    v = pd.DataFrame(volumes, index=idx)
    return MarketData(prices=p, volumes=v, periods_per_year=periods_per_year, name="synthetic")


def load_prices_csv(path: str,
                    date_col: str = "date",
                    volume_col: Optional[str] = None,
                    periods_per_year: int = 252) -> MarketData:
    """从 CSV 加载真实价格（可选，用于把策略跑到自己的数据上）。

    CSV 需含日期列与若干资产价格列；若给 volume_col 则该列作为成交量。
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    volumes = None
    if volume_col and volume_col in df.columns:
        volumes = df[[volume_col]]
        df = df.drop(columns=[volume_col])
    prices = df.apply(pd.to_numeric, errors="coerce").ffill()
    return MarketData(prices=prices, volumes=volumes,
                      periods_per_year=periods_per_year, name="csv")
