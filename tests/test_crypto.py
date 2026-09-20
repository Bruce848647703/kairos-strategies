"""crypto 渠道测试：形状/对齐/有限值、long_only、行和 ≤ 1、确定性 + 行为断言。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.crypto import (
    GridTradingStrategy, DcaStrategy, CryptoMomentum247Strategy, CarryProxyStrategy,
)

ALL_CLASSES = [GridTradingStrategy, DcaStrategy, CryptoMomentum247Strategy, CarryProxyStrategy]
ALL_NAMES = {"grid_trading", "dca", "crypto_momentum_247", "carry_proxy"}
EPS = 1e-9


def _data(prices_dict) -> MarketData:
    prices = pd.DataFrame(prices_dict)
    n = len(next(iter(prices_dict.values())))
    prices.index = pd.bdate_range("2022-01-01", periods=n)
    return MarketData(prices=prices)


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=4, n_days=250, seed=7)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_shape_columns_finite(cls, synth):
    w = cls().generate_weights(synth)
    assert isinstance(w, pd.DataFrame)
    assert w.shape == (len(synth.dates), len(synth.symbols))
    assert list(w.columns) == synth.symbols
    assert (w.index == synth.dates).all()
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_long_only_and_row_sum_cap(cls, synth):
    w = cls().generate_weights(synth)
    assert (w.values >= 0.0).all()
    assert w.abs().sum(axis=1).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_deterministic(cls, synth):
    w1 = cls().generate_weights(synth)
    w2 = cls().generate_weights(synth)
    pd.testing.assert_frame_equal(w1, w2)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_complete(cls):
    s = cls()          # 必须能无参构造
    assert s.channel == "crypto"
    assert s.long_only is True
    assert s.universe in ("timing", "cross_section")
    assert s.name in ALL_NAMES
    assert s.description and s.hypothesis
    assert "加密市场实践" in s.source
    assert isinstance(s.params, dict)


def test_discovered_by_registry():
    found = {s.name for s in discover() if s.channel == "crypto"}
    assert ALL_NAMES <= found


# ---------- 行为断言 ----------

def test_grid_position_grows_as_price_falls():
    """价格逐步下行时，网格仓位应阶梯式递增（低买）。"""
    px = np.linspace(100.0, 70.0, 80)
    data = _data({"A": px})
    w = GridTradingStrategy().generate_weights(data)["A"]
    assert w.iloc[0] == 0.0                      # 起点参考价处空仓
    assert (w.diff().dropna().values >= -EPS).all()   # 下行中仓位单调不减
    assert w.iloc[-1] > w.iloc[0]                # 跌得越深仓位越重
    assert w.iloc[-1] == pytest.approx(1.0)      # 深跌至满格（单资产预算=1）
    assert w.nunique() > 2                       # 阶梯式而非一步到位


def test_grid_stays_flat_when_price_rises():
    """价格单边上行时，网格不追高：仓位保持 0（高卖后空仓）。"""
    px = np.linspace(100.0, 140.0, 60)
    data = _data({"A": px})
    w = GridTradingStrategy().generate_weights(data)["A"]
    assert (w.values == 0.0).all()


def test_dca_weight_monotone_stepwise_accumulation():
    """定投权重随时间单调不减、按 period 阶梯累加、至上限封顶。"""
    n = 240
    data = _data({"A": np.full(n, 100.0)})
    s = DcaStrategy()
    period, tranches = s.params["period"], s.params["n_tranches"]
    w = s.generate_weights(data)["A"]
    assert (w.diff().dropna().values >= -EPS).all()          # 单调不减
    assert w.iloc[0] == pytest.approx(1.0 / tranches)        # 首期买入一份
    assert w.iloc[period] == pytest.approx(2.0 / tranches)   # 一个周期后加一份
    assert w.iloc[(tranches - 1) * period] == pytest.approx(1.0)  # 满份封顶
    assert w.iloc[-1] == 1.0 and w.max() <= 1.0 + EPS
    assert w.iloc[-1] > w.iloc[0]


def test_momentum247_longs_strong_coin_flats_weak_coin():
    """强势上涨币应获得正权重（等预算 1/N），弱势下跌币空仓。"""
    strong = np.linspace(100.0, 300.0, 120)
    weak = np.linspace(100.0, 50.0, 120)
    data = _data({"STRONG": strong, "WEAK": weak})
    w = CryptoMomentum247Strategy().generate_weights(data)
    slow = CryptoMomentum247Strategy.params["slow"]
    assert w["STRONG"].iloc[slow:].min() > 0.0               # 强势币持续持有
    assert w["STRONG"].iloc[-1] == pytest.approx(0.5)        # 等预算 1/2
    assert w["WEAK"].iloc[-1] == 0.0                         # 弱势币空仓


def test_carry_proxy_holds_positive_carry_only():
    """短均线高于长均线（正 carry 代理）时持有，反之空仓。"""
    up = _data({"A": np.linspace(100.0, 200.0, 90)})
    dn = _data({"A": np.linspace(200.0, 100.0, 90)})
    s = CarryProxyStrategy()
    w_up = s.generate_weights(up)["A"]
    w_dn = s.generate_weights(dn)["A"]
    slow = s.params["slow"]
    assert w_up.iloc[slow:].min() > 0.0        # 上行段正 carry，持续持有
    assert w_up.iloc[-1] == pytest.approx(1.0)  # 单资产预算=1
    assert w_dn.iloc[slow:].max() == 0.0       # 下行段负 carry，空仓
