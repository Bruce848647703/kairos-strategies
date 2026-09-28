"""tournament 的离线测试（小型合成数据 + 轻量 dummy 策略，快速且不依赖全部渠道）。"""
import os

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import tournament
from kairos_strategies.base import MarketData, Strategy


class _Const(Strategy):
    channel = "dummy"
    long_only = True
    description = "constant"
    hypothesis = "x"
    source = "test"

    def __init__(self, name, w):
        self.name = name
        self._w = w
        self.params = {}

    def generate_weights(self, data):
        return pd.DataFrame(self._w, index=data.dates, columns=data.symbols)


class _Momentum(Strategy):
    channel = "dummy"
    long_only = True
    description = "momentum"
    hypothesis = "x"
    source = "test"

    def __init__(self, name, lb):
        self.name = name
        self.lb = lb
        self.params = {"lb": lb}

    def generate_weights(self, data):
        mom = data.prices.pct_change(self.lb).fillna(0.0)
        w = (mom > 0).astype(float) / data.n_assets
        return w


def _data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {}
    for k, sym in enumerate(["A", "B", "C", "D"]):
        r = 0.0004 * (k - 1) + 0.01 * rng.standard_normal(n)
        cols[sym] = 100 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252, name="synthetic_test")


def _strategies():
    return [_Const("c_full", 0.25), _Const("c_half", 0.125),
            _Momentum("m10", 10), _Momentum("m60", 60)]


def test_strategy_returns_shape():
    data = _data()
    rets, failed = tournament.strategy_returns(data, _strategies())
    assert set(rets.columns) == {"c_full", "c_half", "m10", "m60"}
    assert len(rets) == len(data.dates)
    assert failed == []


def test_rank_sorted_desc():
    data = _data()
    rets, _ = tournament.strategy_returns(data, _strategies())
    rk = tournament.rank_by_sharpe(rets)
    assert rk["sharpe"].is_monotonic_decreasing
    assert {"name", "sharpe", "total_return", "max_drawdown"} <= set(rk.columns)


def test_correlation_distance_props():
    data = _data()
    rets, _ = tournament.strategy_returns(data, _strategies())
    corr = rets.corr()
    dist = tournament.correlation_distance(corr)
    np.testing.assert_allclose(np.diag(dist.values), 0.0, atol=1e-9)
    assert (dist.values >= -1e-12).all()
    np.testing.assert_allclose(dist.values, dist.values.T, atol=1e-12)


def test_cluster_covers_all():
    data = _data()
    rets, _ = tournament.strategy_returns(data, _strategies())
    clusters = tournament.cluster_strategies(rets.corr(), n_clusters=2)
    flat = [x for v in clusters.values() for x in v]
    assert sorted(flat) == sorted(rets.columns)


def test_select_diversified_respects_thresh_and_k():
    data = _data()
    rets, _ = tournament.strategy_returns(data, _strategies())
    sel = tournament.select_diversified(rets, top_k=3, corr_thresh=0.99)
    assert len(sel) <= 3 and len(set(sel)) == len(sel)
    corr = rets.corr()
    for i, a in enumerate(sel):
        for b in sel[i + 1:]:
            assert abs(corr.loc[a, b]) < 0.99 + 1e-9


def test_combo_returns_finite():
    data = _data()
    rets, _ = tournament.strategy_returns(data, _strategies())
    for m in ("equal", "inverse_vol", "sharpe"):
        c = tournament.combo_returns(rets, list(rets.columns), m)
        assert np.isfinite(c.values).all() and len(c) == len(rets)


def test_run_tournament_writes_outputs(tmp_path):
    data = _data()
    out = tmp_path / "tour"
    res = tournament.run_tournament(data, _strategies(), str(out), top_k=3,
                                    n_clusters=2, chart=False)
    for fn in ("REPORT.md", "ranking.csv", "correlation.csv", "selection.json",
               "combo_equity.csv", "strategy_returns.csv"):
        assert os.path.exists(out / fn), fn
    assert len(res["selected"]) >= 1
    assert "sharpe" in res["combo_metrics"]


def test_deterministic():
    data = _data()
    r1, _ = tournament.strategy_returns(data, _strategies())
    r2, _ = tournament.strategy_returns(data, _strategies())
    pd.testing.assert_frame_equal(r1, r2)
