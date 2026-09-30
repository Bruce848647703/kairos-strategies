"""压力 / 情景稳健性测试用的合成市场情景生成器（纯离线、确定性）。

与 `data.make_synthetic_universe`（把 trend / meanrev / random 三种性格混在同一个 universe）
不同，本模块**每个情景只植入一种明确、可被检出的「市场性格」**，用来检验策略在
单边上涨、崩盘、泡沫、区间震荡、高波动、风格轮动等极端或不利环境下的稳健性。

设计约定：
- 事件型情景（crash / bubble）按**比例**定位事件起点、按「数周」夹取事件长度，
  并把涨跌幅解算成每日漂移，因此 n_days 从 400 到 1000 都能得到同样的市场性格；
- 崩盘/泡沫/高波/轮动都由「共同市场因子 + 各资产 idio 噪声」构成，
  beta 与波动按资产索引确定性生成（不额外消耗随机数），保证同 seed 完全可复现；
- 价格恒正（对数价格 -> exp），成交量随当期涨跌幅放大（压力期放量）。

铁律：100% 原创、只用 numpy/pandas、离线、固定 seed 可复现。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import MarketData

__all__ = ["SCENARIOS", "SCENARIO_DESCRIPTIONS", "describe_scenario", "make_scenario"]

_START = "2018-01-02"

SCENARIOS: List[str] = [
    "trend_up",
    "crash",
    "bubble",
    "choppy",
    "high_vol",
    "sector_rotation",
]

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "trend_up": "全体资产强单边上涨（正漂移 + 轻度正自相关），趋势/动量类策略的主场。",
    "crash": "先平稳上行，中段数周内急跌约 40%，随后部分修复，考验止损与防御能力。",
    "bubble": "涨幅与波动同步加速的指数级吹泡沫，随后断崖式崩塌，追高者重伤。",
    "choppy": "OU 区间震荡市，价格绕中枢来回、收益负自相关，趋势类被反复打脸。",
    "high_vol": "年化波动 60%+ 的高波 GBM 且漂移近 0，仓位与波动率管理决定生死。",
    "sector_rotation": "资产分两组风格轮动：前半段 A 组领涨，后半段 B 组接棒，考验自适应切换。",
}


def describe_scenario(kind: str) -> str:
    """返回某个情景的中文一句话说明；未知 kind 抛 ValueError。"""
    if kind not in SCENARIO_DESCRIPTIONS:
        raise ValueError(f"未知情景 '{kind}'，可选：{', '.join(SCENARIOS)}")
    return SCENARIO_DESCRIPTIONS[kind]


# --------------------------------------------------------------------------- #
# 通用小工具（确定性参数 + 随机路径构造）
# --------------------------------------------------------------------------- #
def _base_params(n_assets: int) -> Tuple[np.ndarray, np.ndarray]:
    """各资产的基准年化波动与起始价（按索引确定性生成，不消耗随机数）。"""
    sig = np.array([0.18 + 0.10 * ((i * 37) % 5) / 4.0 for i in range(n_assets)])
    s0 = np.array([50.0 + 10.0 * (i % 4) for i in range(n_assets)])
    return sig, s0


def _betas(n_assets: int, lo: float = 0.90, hi: float = 1.12) -> np.ndarray:
    """各资产对共同市场因子的暴露：方向一致但强度有差异（确定性）。"""
    if n_assets <= 1:
        return np.ones(n_assets)
    frac = ((np.arange(n_assets) * 7) % 5) / 4.0
    return lo + (hi - lo) * frac


def _phase_bounds(n_days: int, start_frac: float, dur_frac: float,
                  dur_min: int, dur_max: int) -> Tuple[int, int]:
    """事件窗 (起点, 终点)：起点按比例定位，长度按「数周」夹取，保证急跌在数周内完成。"""
    t1 = int(np.clip(round(start_frac * n_days), 1, max(n_days - 2, 1)))
    dur = int(np.clip(round(dur_frac * n_days), dur_min, dur_max))
    t2 = int(min(t1 + dur, n_days))
    return t1, max(t2, t1 + 1)


def _ar1_returns(rng: np.random.Generator, n: int, mu: float, phi: float,
                 sigma: float, dt: float) -> np.ndarray:
    """AR(1) 日收益路径 r_t = mu·dt + phi·r_{t-1} + sigma·√dt·eps（正自相关 = 趋势性格）。"""
    eps = rng.standard_normal(n) * sigma * np.sqrt(dt)
    r = np.zeros(n)
    r[0] = mu * dt + eps[0]
    for t in range(1, n):
        r[t] = mu * dt + phi * r[t - 1] + eps[t]
    return r


def _ou_logprice(rng: np.random.Generator, n: int, log_s0: float, theta: float,
                 sigma: float, dt: float) -> np.ndarray:
    """OU 对数价格路径（区间震荡 / 均值回归性格）：dP = theta·(mu - P)dt + sigma·dW。"""
    eps = rng.standard_normal(n) * sigma * np.sqrt(dt)
    lp = np.zeros(n)
    lp[0] = log_s0
    for t in range(1, n):
        lp[t] = lp[t - 1] + theta * (log_s0 - lp[t - 1]) * dt + eps[t]
    return lp


def _logvol_ar1(rng: np.random.Generator, n: int, level: float, rho: float,
                innov: float) -> np.ndarray:
    """带波动率聚集的年化波动路径：log-vol 走 AR(1)，并做均值校正使中枢为 level。"""
    e = rng.standard_normal(n) * innov
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = rho * lv[t - 1] + e[t]
    var_stat = innov ** 2 / max(1.0 - rho ** 2, 1e-9)
    return level * np.exp(lv - 0.5 * var_stat)


def _from_factor(factor: np.ndarray, rng: np.random.Generator, n_assets: int, n_days: int,
                 s0: np.ndarray, idio_sigma: np.ndarray, dt: float,
                 beta_lo: float = 0.90, beta_hi: float = 1.12) -> np.ndarray:
    """共同因子 + 各资产 idio 噪声 -> (n_days, n_assets) 对数价格矩阵。"""
    betas = _betas(n_assets, beta_lo, beta_hi)
    logp = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        idio = idio_sigma[i] * np.sqrt(dt) * rng.standard_normal(n_days)
        logp[:, i] = np.log(s0[i]) + np.cumsum(betas[i] * factor + idio)
    return logp


def _pack(logp: np.ndarray, idx: pd.DatetimeIndex, kind: str,
          rng: np.random.Generator, periods_per_year: int) -> MarketData:
    """对数价格矩阵 -> MarketData（价格恒正；成交量随当期涨跌幅放大，压力期放量）。"""
    n_assets = logp.shape[1]
    cols = [f"A{i}" for i in range(n_assets)]
    prices = pd.DataFrame(np.exp(np.clip(logp, -50.0, 50.0)), index=idx, columns=cols)
    rets = prices.pct_change().fillna(0.0).values
    base = np.array([1e6 * (1.0 + 0.5 * ((i * 13) % 3)) for i in range(n_assets)])
    shock = 1.0 + 6.0 * np.abs(rets)
    vol = base[None, :] * np.exp(0.3 * rng.standard_normal(rets.shape)) * shock + 1e5
    volumes = pd.DataFrame(vol, index=idx, columns=cols)
    return MarketData(prices=prices, volumes=volumes,
                      periods_per_year=periods_per_year, name=f"scenario:{kind}")


# --------------------------------------------------------------------------- #
# 六个情景
# --------------------------------------------------------------------------- #
def _trend_up(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """全体强单边上涨：共同因子给足正漂移并带轻度正自相关（AR(1)）。"""
    dt = 1.0 / ppy
    sig, s0 = _base_params(n_assets)
    factor = _ar1_returns(rng, n_days, mu=0.42, phi=0.20, sigma=0.16, dt=dt)
    return _from_factor(factor, rng, n_assets, n_days, s0, 0.55 * sig, dt, 0.85, 1.05)


def _crash(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """先平稳上行 -> 中段数周内急跌约 40% -> 随后部分修复（共同因子驱动的同步下跌）。"""
    dt = 1.0 / ppy
    sig, s0 = _base_params(n_assets)
    t1, t2 = _phase_bounds(n_days, 0.45, 0.03, 8, 25)
    t = np.arange(n_days)
    burst = 0.40                                     # 因子层面累计跌幅
    crash_daily = np.log(1.0 - burst) / float(t2 - t1)
    drift = np.where(t < t1, 0.18 * dt, np.where(t < t2, crash_daily, 0.18 * dt))
    volscale = np.where(t < t1, 1.0, np.where(t < t2, 2.2, 1.4))
    factor = drift + volscale * 0.14 * np.sqrt(dt) * rng.standard_normal(n_days)
    return _from_factor(factor, rng, n_assets, n_days, s0, 0.60 * sig, dt)


def _bubble(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """指数级吹泡沫（漂移与波动同步加速）-> 断崖式崩塌 -> 低位横盘。"""
    dt = 1.0 / ppy
    sig, s0 = _base_params(n_assets)
    t1, t2 = _phase_bounds(n_days, 0.55, 0.03, 6, 15)
    t = np.arange(n_days)
    runup, burst = 4.0, 0.60                         # 吹泡涨到 4 倍，随后崩掉 60%
    years1 = max(t1, 1) / ppy
    mu0 = 0.10
    k = 3.0 * (np.log(runup) / years1 - mu0)         # 解算加速度使累计涨幅恰为 runup
    frac = np.clip(t / float(max(t1, 1)), 0.0, 1.0)
    blowoff = (mu0 + k * frac ** 2) * dt
    drift = np.where(t < t1, blowoff,
                     np.where(t < t2, np.log(1.0 - burst) / float(t2 - t1), 0.05 * dt))
    sigm = 0.13
    vol = np.where(t < t1, sigm * (1.0 + 1.8 * frac ** 2),
                   np.where(t < t2, sigm * 3.5, sigm * 1.5))
    factor = drift + vol * np.sqrt(dt) * rng.standard_normal(n_days)
    return _from_factor(factor, rng, n_assets, n_days, s0, 0.55 * sig, dt)


def _choppy(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """区间震荡：各资产独立 OU 快速回归自身中枢，无趋势、收益负自相关。"""
    dt = 1.0 / ppy
    _, s0 = _base_params(n_assets)
    logp = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        theta = 24.0 + 6.0 * ((i * 7) % 3) / 2.0     # 24~30：快速回归（收益负自相关）
        sigma = 0.46 + 0.06 * ((i * 13) % 3) / 2.0   # 0.46~0.52：与 theta 匹配出 ±7% 箱体
        logp[:, i] = _ou_logprice(rng, n_days, float(np.log(s0[i])), theta, sigma, dt)
    return logp


def _zero_drift(rng: np.random.Generator, n_days: int, vol: np.ndarray, dt: float) -> np.ndarray:
    """零漂移的高波收益流：波动项去均值（抹掉实现漂移）+ 波动拖累项 -σ²/2·dt。

    这样「漂移近 0」在任何样本长度下都成立——否则纯随机游走的噪声会在长样本上
    把「高波无方向」误演成单边熊市，情景性格就被稀释了。
    """
    noise = vol * np.sqrt(dt) * rng.standard_normal(n_days)
    return (noise - noise.mean()) - 0.5 * vol ** 2 * dt


def _high_vol(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """高波动 GBM：共同因子年化波动 ~50%（叠加 idio 后单资产 60%+），漂移近 0。"""
    dt = 1.0 / ppy
    _, s0 = _base_params(n_assets)
    sigf = _logvol_ar1(rng, n_days, level=0.50, rho=0.90, innov=0.12)
    factor = _zero_drift(rng, n_days, sigf, dt)
    idio_sigma = 0.30 + 0.18 * ((np.arange(n_assets) * 13) % 4) / 3.0   # 0.30~0.48
    betas = _betas(n_assets, 0.90, 1.10)
    logp = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        idio = _zero_drift(rng, n_days, np.full(n_days, idio_sigma[i]), dt)
        logp[:, i] = np.log(s0[i]) + np.cumsum(betas[i] * factor + idio)
    return logp


def _sector_rotation(rng: np.random.Generator, n_assets: int, n_days: int, ppy: float) -> np.ndarray:
    """风格轮动：前 n_a 只为 A 组（前半段领涨），其余为 B 组（后半段接棒）。"""
    dt = 1.0 / ppy
    _, s0 = _base_params(n_assets)
    half = max(n_days // 2, 1)
    n_a = max((n_assets + 1) // 2, 1)
    t = np.arange(n_days)
    mkt = 0.05 * dt + 0.10 * np.sqrt(dt) * rng.standard_normal(n_days)   # 弱共同因子
    logp = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        lead = i < n_a
        mu_pre = 0.55 if lead else -0.05
        mu_post = -0.10 if lead else 0.55
        drift = np.where(t < half, mu_pre * dt, mu_post * dt)
        idio = 0.16 * np.sqrt(dt) * rng.standard_normal(n_days)
        logp[:, i] = np.log(s0[i]) + np.cumsum(drift + 0.5 * mkt + idio)
    return logp


_BUILDERS: Dict[str, Callable[[np.random.Generator, int, int, float], np.ndarray]] = {
    "trend_up": _trend_up,
    "crash": _crash,
    "bubble": _bubble,
    "choppy": _choppy,
    "high_vol": _high_vol,
    "sector_rotation": _sector_rotation,
}


def make_scenario(kind: str, n_assets: int = 8, n_days: int = 1000, seed: int = 2026,
                  periods_per_year: int = 252) -> MarketData:
    """生成指定「市场性格」的合成情景（价格 + 成交量），固定 seed 完全可复现。

    Parameters
    ----------
    kind : 情景名，取值见 `SCENARIOS`
        trend_up / crash / bubble / choppy / high_vol / sector_rotation
    n_assets : 资产数（列名 A0..A{n-1}）
    n_days : 交易日数（事件型情景按比例定位，长度变化不改变性格）
    seed : 随机种子
    periods_per_year : 年化周期数（默认 252）

    Returns
    -------
    MarketData：prices / volumes 形状均为 (n_days, n_assets)，价格恒正、无 NaN。
    """
    if kind not in _BUILDERS:
        raise ValueError(f"未知情景 '{kind}'，可选：{', '.join(SCENARIOS)}")
    n_assets = int(max(1, n_assets))
    n_days = int(max(20, n_days))
    rng = np.random.default_rng(int(seed))
    logp = _BUILDERS[kind](rng, n_assets, n_days, float(periods_per_year))
    idx = pd.bdate_range(start=_START, periods=n_days)
    return _pack(logp, idx, kind, rng, int(periods_per_year))
