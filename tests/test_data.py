import numpy as np
import pandas as pd

from kairos_strategies import make_synthetic_universe


def test_shape_and_index():
    d = make_synthetic_universe(n_assets=6, n_days=300, seed=2)
    assert d.prices.shape == (300, 6)
    assert d.n_assets == 6
    assert isinstance(d.dates, pd.DatetimeIndex)
    assert d.volumes is not None and d.volumes.shape == d.prices.shape


def test_prices_positive():
    d = make_synthetic_universe(n_assets=8, n_days=500, seed=3)
    assert (d.prices.values > 0).all()


def test_deterministic_same_seed():
    a = make_synthetic_universe(n_assets=4, n_days=200, seed=42)
    b = make_synthetic_universe(n_assets=4, n_days=200, seed=42)
    pd.testing.assert_frame_equal(a.prices, b.prices)


def test_different_seed_differs():
    a = make_synthetic_universe(n_assets=4, n_days=200, seed=1)
    b = make_synthetic_universe(n_assets=4, n_days=200, seed=2)
    assert not np.allclose(a.prices.values, b.prices.values)


def test_regimes_differ_in_autocorrelation():
    # trend 资产收益应呈正自相关，meanrev 资产应呈负自相关（植入的性格可被检出）
    d = make_synthetic_universe(n_assets=6, n_days=1500, seed=7)
    rets = d.returns()
    ac = {c: rets[c].autocorr(1) for c in rets.columns}
    trend_syms = [c for i, c in enumerate(rets.columns) if i % 3 == 0]
    meanrev_syms = [c for i, c in enumerate(rets.columns) if i % 3 == 1]
    assert np.mean([ac[s] for s in trend_syms]) > 0
    assert np.mean([ac[s] for s in meanrev_syms]) < np.mean([ac[s] for s in trend_syms])
