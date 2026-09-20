import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.factor import (
    IdioMomentumStrategy,
    LowVolatilityStrategy,
    ShortTermReversalStrategy,
    TrendQualityStrategy,
)

FACTOR_CLASSES = [
    LowVolatilityStrategy,
    ShortTermReversalStrategy,
    TrendQualityStrategy,
    IdioMomentumStrategy,
]

EPS = 1e-9


def _synth(prices_dict):
    prices = pd.DataFrame(prices_dict)
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    return MarketData(prices=prices)


# ---------------------------- 通用契约 ----------------------------

@pytest.mark.parametrize("cls", FACTOR_CLASSES)
def test_shape_columns_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == data.symbols
    assert len(w) == len(data.dates)
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", FACTOR_CLASSES)
def test_long_only_and_row_sum_le_one(cls):
    data = make_synthetic_universe(n_assets=7, n_days=300, seed=9)
    w = cls().generate_weights(data)
    assert (w.values >= 0.0).all()
    assert w.values.sum(axis=1).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", FACTOR_CLASSES)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=250, seed=3)
    a = cls().generate_weights(data)
    b = cls().generate_weights(data)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", FACTOR_CLASSES)
def test_fully_invested_after_warmup(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert abs(w.iloc[-1].sum() - 1.0) < EPS


@pytest.mark.parametrize("cls", FACTOR_CLASSES)
def test_meta(cls):
    s = cls()
    m = s.meta()
    assert m["channel"] == "factor"
    assert m["universe"] == "cross_section"
    assert s.long_only is True
    for k in ("name", "description", "hypothesis", "source", "params"):
        assert m[k]


def test_discover_finds_factor_channel():
    names = {s.name for s in discover()}
    assert {"low_volatility", "short_term_reversal",
            "trend_quality", "idio_momentum"} <= names


# ---------------------------- 行为断言 ----------------------------

def test_low_volatility_gives_least_volatile_highest_weight():
    rng = np.random.default_rng(0)
    n = 200
    cols = {"LOW": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))}  # 极低波动
    for i in range(5):
        cols[f"H{i}"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))  # 高波动
    w = LowVolatilityStrategy().generate_weights(_synth(cols))
    last = w.iloc[-1]
    assert last.idxmax() == "LOW"
    assert last["LOW"] > 0
    assert last["LOW"] > last.drop("LOW").max()


def test_short_term_reversal_gives_recent_loser_high_weight():
    rng = np.random.default_rng(2)
    n = 60
    base = np.linspace(100, 110, n)
    cols = {f"U{i}": base + rng.normal(0, 0.3, n) for i in range(5)}  # 温和上涨
    cols["DROP"] = np.concatenate([np.full(n - 5, 100.0),
                                   np.linspace(100, 70, 5)])           # 近期大跌
    w = ShortTermReversalStrategy().generate_weights(_synth(cols))
    last = w.iloc[-1]
    assert last.idxmax() == "DROP"
    assert last["DROP"] > 0
    assert last["DROP"] > last.drop("DROP").max()


def test_trend_quality_gives_smooth_uptrend_high_weight():
    rng = np.random.default_rng(1)
    n = 80
    t = np.arange(n)
    cols = {"SMOOTH": np.linspace(100, 160, n)}  # 单边平滑上涨（效率比≈1）
    for i in range(5):
        cols[f"C{i}"] = 100 + 6 * np.sin(t * (0.3 + 0.1 * i)) + rng.normal(0, 0.3, n)  # 震荡
    w = TrendQualityStrategy().generate_weights(_synth(cols))
    last = w.iloc[-1]
    assert last.idxmax() == "SMOOTH"
    assert last["SMOOTH"] > 0
    assert last["SMOOTH"] > last.drop("SMOOTH").max()


def test_idio_momentum_gives_relative_outperformer_high_weight():
    n = 120
    t = np.arange(n)
    # WIN 相对市场（其余资产均值）显著跑赢；其余资产同向温和走势
    cols = {"WIN": 100 * np.exp(np.linspace(0, 0.40, n))}
    for i in range(5):
        cols[f"M{i}"] = 100 * np.exp(np.linspace(0, 0.02, n) + 0.01 * np.sin(t * 0.2 + i))
    w = IdioMomentumStrategy().generate_weights(_synth(cols))
    last = w.iloc[-1]
    assert last.idxmax() == "WIN"
    assert last["WIN"] > 0
