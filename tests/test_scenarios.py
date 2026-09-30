"""情景生成器测试：形状/正价/确定性 + 每种情景植入的「市场性格」必须可被检出。

全部离线、确定性，用较小的 n_days 保证快速。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import metrics
from kairos_strategies.base import MarketData
from kairos_strategies.engine import Backtester
from kairos_strategies.scenarios import (
    SCENARIO_DESCRIPTIONS,
    SCENARIOS,
    describe_scenario,
    make_scenario,
)

N_DAYS = 400
N_ASSETS = 8
SEED = 2026


def _buy_hold(data: MarketData):
    """等权买入持有（期初等权、不再平衡）的净值与收益流。"""
    eq = (data.prices / data.prices.iloc[0]).mean(axis=1)
    return eq, eq.pct_change().fillna(0.0)


def _mean_autocorr(data: MarketData, lag: int = 1) -> float:
    """各资产 lag 阶收益自相关的均值（跨资产平均以压低抽样噪声）。"""
    rets = data.returns()
    return float(np.mean([rets[c].autocorr(lag) for c in rets.columns]))


# --------------------------------------------------------------------------- #
# 元信息一致性
# --------------------------------------------------------------------------- #
def test_scenarios_list_matches_descriptions():
    assert SCENARIOS == ["trend_up", "crash", "bubble", "choppy", "high_vol", "sector_rotation"]
    assert set(SCENARIO_DESCRIPTIONS) == set(SCENARIOS)
    for kind in SCENARIOS:
        text = describe_scenario(kind)
        assert isinstance(text, str) and len(text) >= 8
        assert text == SCENARIO_DESCRIPTIONS[kind]


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        describe_scenario("no_such_regime")
    with pytest.raises(ValueError):
        make_scenario("no_such_regime", n_days=N_DAYS)


# --------------------------------------------------------------------------- #
# 通用契约：形状 / 恒正 / 无 NaN / 确定性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", SCENARIOS)
def test_shape_positive_and_finite(kind):
    d = make_scenario(kind, n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    assert isinstance(d, MarketData)
    assert d.prices.shape == (N_DAYS, N_ASSETS)
    assert d.n_assets == N_ASSETS and len(d.dates) == N_DAYS
    assert isinstance(d.dates, pd.DatetimeIndex)
    assert d.symbols == [f"A{i}" for i in range(N_ASSETS)]
    assert np.isfinite(d.prices.values).all()
    assert (d.prices.values > 0).all()
    assert d.name == f"scenario:{kind}"
    assert d.periods_per_year == 252


@pytest.mark.parametrize("kind", SCENARIOS)
def test_volumes_attached_and_positive(kind):
    d = make_scenario(kind, n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    assert d.volumes is not None
    assert d.volumes.shape == d.prices.shape
    assert list(d.volumes.columns) == d.symbols
    assert np.isfinite(d.volumes.values).all() and (d.volumes.values > 0).all()


@pytest.mark.parametrize("kind", SCENARIOS)
def test_deterministic_same_seed(kind):
    a = make_scenario(kind, n_assets=6, n_days=N_DAYS, seed=SEED)
    b = make_scenario(kind, n_assets=6, n_days=N_DAYS, seed=SEED)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    pd.testing.assert_frame_equal(a.volumes, b.volumes)


@pytest.mark.parametrize("kind", SCENARIOS)
def test_different_seed_differs(kind):
    a = make_scenario(kind, n_assets=6, n_days=N_DAYS, seed=1)
    b = make_scenario(kind, n_assets=6, n_days=N_DAYS, seed=2)
    assert not np.allclose(a.prices.values, b.prices.values)


@pytest.mark.parametrize("kind", SCENARIOS)
def test_returns_api_clean(kind):
    d = make_scenario(kind, n_assets=4, n_days=N_DAYS, seed=SEED)
    rets = d.returns()
    assert rets.shape == d.prices.shape
    assert np.isfinite(rets.values).all()
    assert d.log_returns().notna().all().all()
    vol = d.rolling_vol(20).iloc[25:]
    assert np.isfinite(vol.values).all() and (vol.values > 0).all()
    # 单日不出现 -100%（价格恒正的必然结果）
    assert rets.values.min() > -1.0


@pytest.mark.parametrize("kind", SCENARIOS)
def test_engine_can_backtest_every_scenario(kind):
    """引擎兼容性冒烟：等权常量权重在每个情景上都能回测出有限指标。"""
    d = make_scenario(kind, n_assets=4, n_days=N_DAYS, seed=SEED)
    w = pd.DataFrame(1.0 / d.n_assets, index=d.dates, columns=d.symbols)
    res = Backtester(cost_rate=0.001, periods_per_year=d.periods_per_year).run(d, w)
    m = res.metrics
    assert np.isfinite(res.returns.values).all()
    assert len(res.returns) == N_DAYS
    for key in ("sharpe", "total_return", "max_drawdown", "volatility"):
        assert np.isfinite(m[key])
    assert m["max_drawdown"] >= 0.0


# --------------------------------------------------------------------------- #
# 性格可检出
# --------------------------------------------------------------------------- #
def test_trend_up_is_strong_one_way_rally():
    d = make_scenario("trend_up", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    _, r = _buy_hold(d)
    assert metrics.total_return(r) > 0.30          # 显著为正
    assert metrics.sharpe(r) > 1.0                 # 买入持有夏普高
    assert _mean_autocorr(d) > 0.02                # 轻度正自相关（趋势性格）
    # 每个资产都是上涨的（"全体"单边）
    assert (d.prices.iloc[-1] > d.prices.iloc[0]).all()


def test_crash_has_deep_drawdown_after_calm_start():
    d = make_scenario("crash", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    eq, r = _buy_hold(d)
    assert metrics.max_drawdown(r) > 0.30          # 深回撤
    pre = eq.iloc[int(0.40 * N_DAYS)] / eq.iloc[0] - 1.0
    assert pre > 0.0                               # 崩盘前平稳/上行
    worst_3w = float(eq.pct_change(15).min())
    assert worst_3w < -0.25                        # 数周内急跌
    trough = int(eq.values.argmin())
    assert trough > int(0.30 * N_DAYS)             # 低点在中段而非开头
    assert eq.iloc[-1] > eq.iloc[trough]           # 随后部分修复


def test_bubble_blows_up_then_collapses():
    d = make_scenario("bubble", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    eq, r = _buy_hold(d)
    peak = int(eq.values.argmax())
    runup = eq.iloc[peak] / eq.iloc[0]
    assert runup > 2.0                             # 先大涨（吹泡）
    assert 0.10 < peak / N_DAYS < 0.95             # 峰在中后段，不是开局即峰
    assert metrics.max_drawdown(r) > 0.40          # 峰值后断崖式回撤
    after = eq.iloc[peak:]
    assert after.min() / eq.iloc[peak] - 1.0 < -0.40   # 峰后腰斩以上
    assert int(after.values.argmin()) > 0            # 低点确在峰值之后（后大跌）
    assert float(eq.pct_change(10).min()) < -0.25  # 崩塌在数日内完成


def test_choppy_is_range_bound_mean_reverting():
    d = make_scenario("choppy", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    _, r = _buy_hold(d)
    assert abs(metrics.total_return(r)) < 0.15     # 累计收益接近 0
    assert _mean_autocorr(d) < -0.01               # 收益负自相关（均值回归）
    # 箱体：对数价格围绕自身均值的离散度小，且终点回到起点附近
    logp = np.log(d.prices)
    assert float(logp.sub(logp.mean()).std().mean()) < 0.15
    assert float(np.abs(logp.iloc[-1] - logp.iloc[0]).mean()) < 0.20
    assert metrics.max_drawdown(r) < 0.35          # 震荡市不该有崩盘级回撤


def test_high_vol_is_volatile_but_driftless():
    d = make_scenario("high_vol", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    _, r = _buy_hold(d)
    assert metrics.volatility(r) > 0.40            # 组合年化波动 > 40%
    per_asset = d.returns().std() * np.sqrt(252)
    assert float(per_asset.min()) > 0.50           # 单资产年化波动 60% 量级
    assert abs(r.mean()) / r.std() < 0.10          # 漂移相对波动可忽略（漂移近 0）
    assert abs(_mean_autocorr(d)) < 0.10           # 无趋势性自相关
    assert metrics.max_drawdown(r) > 0.25          # 高波必然带来深回撤


def test_sector_rotation_switches_leaders():
    d = make_scenario("sector_rotation", n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
    half = N_DAYS // 2
    n_a = (N_ASSETS + 1) // 2                      # A 组 = 前 n_a 只
    p = d.prices
    first = p.iloc[half - 1] / p.iloc[0] - 1.0
    second = p.iloc[-1] / p.iloc[half - 1] - 1.0
    lead_first = int(np.argmax(first.values))
    lead_second = int(np.argmax(second.values))
    assert lead_first < n_a <= lead_second         # 前后半段最强资产分属两组
    assert lead_first != lead_second
    assert float(first.iloc[:n_a].mean()) > float(first.iloc[n_a:].mean())
    assert float(second.iloc[n_a:].mean()) > float(second.iloc[:n_a].mean())


def test_scenarios_are_mutually_distinct():
    """不同情景的等权买入持有特征不应雷同（性格确实被植入了）。"""
    feats = {}
    for kind in SCENARIOS:
        d = make_scenario(kind, n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)
        _, r = _buy_hold(d)
        feats[kind] = (metrics.total_return(r), metrics.max_drawdown(r), metrics.volatility(r))
    for a in SCENARIOS:
        for b in SCENARIOS:
            if a >= b:
                continue
            assert not np.allclose(feats[a], feats[b], atol=1e-6)
