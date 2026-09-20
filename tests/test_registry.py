import numpy as np
import pandas as pd

from kairos_strategies import by_channel, discover, make_synthetic_universe
from kairos_strategies.base import Strategy


def test_discover_finds_technical():
    names = {s.name for s in discover()}
    assert {"sma_cross", "rsi_reversion", "macd_trend",
            "bollinger_breakout", "donchian_turtle"} <= names


def test_discover_unique_names_and_instances():
    strat = discover()
    names = [s.name for s in strat]
    assert len(names) == len(set(names))
    assert all(isinstance(s, Strategy) for s in strat)


def test_by_channel_groups():
    grouped = by_channel(discover())
    assert "technical" in grouped
    assert all(s.channel == "technical" for s in grouped["technical"])


def test_every_strategy_meta_and_weights_shape():
    data = make_synthetic_universe(n_assets=5, n_days=200, seed=11)
    for s in discover():
        meta = s.meta()
        for k in ("name", "channel", "description", "hypothesis", "source", "params"):
            assert k in meta
        w = s.generate_weights(data)
        assert isinstance(w, pd.DataFrame)
        assert list(w.columns) == data.symbols
        assert len(w) == len(data.dates)
        assert np.isfinite(w.values).all()


def test_every_strategy_weights_bounded():
    data = make_synthetic_universe(n_assets=6, n_days=250, seed=13)
    for s in discover():
        w = s.generate_weights(data)
        if s.long_only:
            assert (w.values >= -1e-12).all(), s.name
        assert (np.abs(w.values) <= 1.0 + 1e-9).all(), s.name


def test_strategies_deterministic():
    data = make_synthetic_universe(n_assets=4, n_days=150, seed=17)
    for s in discover():
        a = s.generate_weights(data)
        b = s.generate_weights(data)
        pd.testing.assert_frame_equal(a, b)
