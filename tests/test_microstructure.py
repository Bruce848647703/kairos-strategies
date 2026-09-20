import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.microstructure import (
    LiquidityPremiumStrategy,
    VolumeImbalanceStrategy,
    VolumePriceDivergenceStrategy,
    VwapReversionStrategy,
)

MS_CLASSES = [
    VolumeImbalanceStrategy,
    VwapReversionStrategy,
    VolumePriceDivergenceStrategy,
    LiquidityPremiumStrategy,
]

EXPECTED_NAMES = {
    "volume_imbalance",
    "vwap_reversion",
    "volume_price_divergence",
    "liquidity_premium",
}

EPS = 1e-9


def _synth(prices_dict, volumes_dict=None):
    prices = pd.DataFrame(prices_dict)
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    volumes = None
    if volumes_dict is not None:
        volumes = pd.DataFrame(volumes_dict, index=prices.index)
    return MarketData(prices=prices, volumes=volumes)


# ---------------------------- 通用契约 ----------------------------

@pytest.mark.parametrize("cls", MS_CLASSES)
def test_shape_columns_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == data.symbols
    assert len(w) == len(data.dates)
    assert w.index.equals(data.prices.index)
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", MS_CLASSES)
def test_long_only_and_row_sum_le_one(cls):
    data = make_synthetic_universe(n_assets=7, n_days=300, seed=9)
    w = cls().generate_weights(data)
    assert (w.values >= 0.0).all()
    assert w.values.sum(axis=1).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", MS_CLASSES)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=250, seed=3)
    a = cls().generate_weights(data)
    b = cls().generate_weights(data)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", MS_CLASSES)
def test_volumes_none_fallback_no_error(cls):
    """volumes=None 时优雅回退到纯价格逻辑：不报错且契约不变。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=11)
    dp = MarketData(prices=data.prices.copy(), volumes=None,
                    periods_per_year=data.periods_per_year, name="no_volume")
    assert dp.volumes is None
    w = cls().generate_weights(dp)
    assert list(w.columns) == dp.symbols
    assert len(w) == len(dp.dates)
    assert np.isfinite(w.values).all()
    assert (w.values >= 0.0).all()
    assert w.values.sum(axis=1).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", MS_CLASSES)
def test_meta(cls):
    s = cls()
    m = s.meta()
    assert m["channel"] == "microstructure"
    assert m["universe"] in ("timing", "cross_section")
    assert s.long_only is True
    for k in ("name", "description", "hypothesis", "source", "params"):
        assert m[k]


def test_names_unique_and_expected():
    names = [cls().name for cls in MS_CLASSES]
    assert set(names) == EXPECTED_NAMES
    assert len(names) == len(set(names)) == 4


def test_constructible_without_args():
    for cls in MS_CLASSES:
        assert isinstance(cls(), cls)


# ---------------------------- 行为断言 ----------------------------

def test_volume_imbalance_longs_up_volume_dominance():
    """每天上涨（量全记在上涨日）→ 满预算持有；每天下跌 → 清仓。"""
    n = 100
    t = np.arange(n)
    prices = {"UP": 100.0 * np.exp(0.004 * t),
              "DOWN": 100.0 * np.exp(-0.004 * t)}
    volumes = {"UP": np.full(n, 1e6), "DOWN": np.full(n, 1e6)}
    w = VolumeImbalanceStrategy().generate_weights(_synth(prices, volumes))
    last = w.iloc[-1]
    assert last["UP"] == pytest.approx(0.5)      # 1/2 等预算，买压失衡比≈+1
    assert last["DOWN"] == 0.0                   # 卖压失衡比≈-1，离场


def test_vwap_reversion_longs_below_vwap_and_exits_above():
    """价格持续低于滚动 VWAP → 做多；持续高于 VWAP → 离场（权重 0）。"""
    n = 120
    t = np.arange(n)
    prices = {"BELOW": 100.0 * np.exp(-0.008 * t),   # 单边下行，价 < VWAP
              "ABOVE": 100.0 * np.exp(+0.008 * t)}   # 单边上行，价 > VWAP
    volumes = {"BELOW": np.full(n, 1e6), "ABOVE": np.full(n, 1e6)}
    w = VwapReversionStrategy().generate_weights(_synth(prices, volumes))
    last = w.iloc[-1]
    assert last["BELOW"] == pytest.approx(0.5)   # 折价超进入带 → 持有
    assert last["ABOVE"] == 0.0                  # 溢价于 VWAP → 不持有


def test_volume_price_divergence_prefers_confirmed_breakout():
    """同样价涨创新高：放量确认给更高权重，缩量背离清仓。"""
    n = 150
    t = np.arange(n)
    price = 100.0 * np.exp(0.006 * t)                # 两资产同步上行创新高
    prices = {"CONF": price, "DIV": price.copy()}
    volumes = {"CONF": np.linspace(5e5, 2e6, n),      # 价涨量增（确认）
               "DIV": np.linspace(2e6, 5e5, n)}       # 价涨量缩（背离）
    w = VolumePriceDivergenceStrategy().generate_weights(_synth(prices, volumes))
    last = w.iloc[-1]
    assert last["CONF"] > last["DIV"]
    assert last["CONF"] > 0.0
    assert last["DIV"] == 0.0                         # 新高+缩量 → 背离清仓


def test_liquidity_premium_prefers_low_volume_asset():
    """低量且缩量资产应获得篮子内最高权重，且整行归一到 ≈1。"""
    n = 150
    t = np.arange(n)
    prices = {"LOW": 100.0 * np.exp(0.0005 * t)}
    volumes = {"LOW": np.linspace(8e5, 2e5, n)}       # 低量且相对自身缩量
    for i in range(5):
        prices[f"H{i}"] = 100.0 * np.exp(0.0005 * t)
        volumes[f"H{i}"] = np.linspace(2e6, 6e6, n) * (1.0 + 0.2 * i)  # 高量放量
    w = LiquidityPremiumStrategy().generate_weights(_synth(prices, volumes))
    last = w.iloc[-1]
    assert last.idxmax() == "LOW"
    assert last["LOW"] > 0.0
    assert last["LOW"] > last.drop("LOW").max()
    assert last.sum() == pytest.approx(1.0)
