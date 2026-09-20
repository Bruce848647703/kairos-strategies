"""绩效指标（自包含，纯 numpy/pandas）。输入为每期收益率序列。"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    years = n / float(periods_per_year)
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else 0.0


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd < 1e-12 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    down = excess[excess < 0]
    if down.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    dvol = float(np.sqrt((down ** 2).sum() / max(len(r) - 1, 1)))
    return float(excess.mean() / dvol * np.sqrt(periods_per_year)) if dvol > 1e-12 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(-dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, periods_per_year) / mdd) if mdd > 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    active = r[r != 0]
    return float((active > 0).sum() / len(active)) if len(active) else 0.0


def profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize(returns: pd.Series, turnover: pd.Series,
              risk_free: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
    """一次性汇总常用指标。"""
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "volatility": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, risk_free, periods_per_year),
        "sortino": sortino(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "total_turnover": float(turnover.sum()) if len(turnover) else 0.0,
        "n_periods": float(len(returns.dropna())),
    }
