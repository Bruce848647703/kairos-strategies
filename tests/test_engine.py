import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, make_synthetic_universe


def _simple():
    idx = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    return MarketData(prices=prices)


def test_no_lookahead_uses_previous_weight():
    data = _simple()
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=data.dates)
    res = Backtester(cost_rate=0.0).run(data, w)
    # 第 1 期目标=1，但当期持有=上一期=0 -> 净收益 0
    assert res.returns.iloc[1] == pytest.approx(0.0, abs=1e-12)
    # 第 2 期才享受 A 的 +10%
    assert res.returns.iloc[2] == pytest.approx(0.1, rel=1e-9)


def test_full_investment_tracks_asset_zero_cost():
    data = _simple()
    w = pd.DataFrame(1.0, index=data.dates, columns=["A"])
    res = Backtester(cost_rate=0.0).run(data, w)
    asset_ret = data.prices["A"].pct_change().fillna(0.0)
    pd.testing.assert_series_equal(res.returns.iloc[1:].rename(None),
                                   asset_ret.iloc[1:].rename(None), check_names=False)


def test_costs_reduce_returns():
    data = make_synthetic_universe(n_assets=4, n_days=200, seed=1)
    w = pd.DataFrame(0.25, index=data.dates, columns=data.symbols)
    free = Backtester(cost_rate=0.0).run(data, w).returns.sum()
    costly = Backtester(cost_rate=0.002).run(data, w).returns.sum()
    assert costly <= free + 1e-12


def test_first_period_turnover_is_initial_build():
    data = _simple()
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=data.dates)
    res = Backtester(cost_rate=0.0).run(data, w)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_metrics_present_and_equity_starts_near_one():
    data = make_synthetic_universe(n_assets=3, n_days=120, seed=5)
    w = pd.DataFrame(1 / 3, index=data.dates, columns=data.symbols)
    res = Backtester().run(data, w)
    for k in ("total_return", "sharpe", "max_drawdown", "cagr"):
        assert k in res.metrics
    assert res.equity.iloc[0] == pytest.approx(1.0 + res.returns.iloc[0])
