"""volatility 渠道测试：形状/边界/确定性 + 行为断言（高波降仓、突破持有、低波多配）。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.volatility import (
    AtrBreakout,
    VolRegimeFilter,
    VolScaledMomentum,
    VolTargetOverlay,
    _close_atr,
    _market_returns,
    _realized_vol,
)

EPS = 1e-9
ALL = [VolTargetOverlay, AtrBreakout, VolRegimeFilter, VolScaledMomentum]
NAMES = {"vol_target", "atr_breakout", "vol_regime_filter", "vol_scaled_momentum"}


# ---------------------------------------------------------------- 数据构造
def _regime_market(n_low: int = 340, n_high: int = 60, low_sigma: float = 0.002,
                   high_sigma: float = 0.050, seed: int = 7, n_assets: int = 4,
                   drift: float = 0.0003):
    """前 n_low 天低波动、后 n_high 天高波动的合成行情（确定性）。"""
    rng = np.random.default_rng(seed)
    n = n_low + n_high
    sigma = np.concatenate([np.full(n_low, low_sigma), np.full(n_high, high_sigma)])
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        r = drift + rng.standard_normal(n) * sigma
        cols["A%d" % i] = 100.0 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_low


def _series_market(values, name: str = "A0"):
    px = pd.DataFrame({name: np.asarray(values, dtype=float)},
                      index=pd.bdate_range("2020-01-01", periods=len(values)))
    return MarketData(prices=px, periods_per_year=252)


def _exposure(w: pd.DataFrame) -> pd.Series:
    return w.sum(axis=1)


# ---------------------------------------------------------------- 通用契约
@pytest.mark.parametrize("cls", ALL)
def test_shape_columns_index_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert w.shape == (len(data.dates), len(data.symbols))
    assert list(w.columns) == data.symbols
    assert w.index.equals(data.dates)
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", ALL)
def test_long_only_and_row_sum_cap(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    w = cls().generate_weights(data)
    assert (w.values >= 0.0).all()
    assert np.abs(w.values).sum(axis=1).max() <= 1.0 + EPS
    assert np.abs(w.values).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=3)
    s = cls()
    pd.testing.assert_frame_equal(s.generate_weights(data), s.generate_weights(data))


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                       # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "volatility"
    assert m["long_only"] is True
    assert m["name"] in NAMES
    assert m["universe"] in {"timing", "cross_section", "overlay"}
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_discovered_with_unique_names():
    got = {s.name: s for s in discover() if s.channel == "volatility"}
    assert NAMES <= set(got)
    assert len({s.name for s in discover()}) == len(discover())


@pytest.mark.parametrize("cls", ALL)
def test_no_lookahead_future_prices_do_not_change_history(cls):
    """改动 t 之后的价格，t 及之前的权重必须完全不变（自证无未来函数）。"""
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=5)
    t = 200
    base = cls().generate_weights(data)
    p2 = data.prices.copy()
    ramp = np.linspace(1.0, 2.5, len(p2) - t - 1)[:, None]
    p2.iloc[t + 1:] = p2.iloc[t + 1:].values * ramp
    mutated = cls().generate_weights(MarketData(prices=p2, periods_per_year=252))
    pd.testing.assert_frame_equal(base.iloc[:t + 1], mutated.iloc[:t + 1])


# ---------------------------------------------------------------- 行为断言
def test_vol_target_full_in_low_vol_and_deleveraged_in_high_vol():
    data, n_low = _regime_market()
    w = VolTargetOverlay().generate_weights(data)
    expo = _exposure(w)
    low = expo.iloc[60:n_low].mean()
    high = expo.iloc[n_low + 20:].mean()
    assert high < low                                   # 高波段总敞口更低
    assert low == pytest.approx(1.0, abs=1e-6)          # 低波段：缩放触顶 -> 满仓（不加杠杆）
    assert high < 0.5                                   # 高波段显著降杠杆
    assert expo.max() <= 1.0 + EPS


def test_vol_target_scale_equals_target_over_realized_vol():
    data, n_low = _regime_market()
    p = VolTargetOverlay.params
    rv = _realized_vol(_market_returns(data), p["window"], data.periods_per_year)
    expected = np.minimum(p["target_vol"] / np.maximum(rv, p["vol_floor"]), p["max_scale"])
    expo = _exposure(VolTargetOverlay().generate_weights(data))
    ok = rv.notna()
    assert np.allclose(expo[ok].values, expected[ok].values, atol=1e-12)
    assert (expo[~ok].values == 0.0).all()              # 波动未知时不建仓


def test_vol_regime_filter_risk_off_in_high_vol():
    data, n_low = _regime_market()
    w = VolRegimeFilter().generate_weights(data)
    expo = _exposure(w)
    low = expo.iloc[200:n_low].mean()
    high = expo.iloc[n_low + 20:].mean()
    assert low > 0.4                                    # 低波 regime 大部分时间持有
    assert high < low
    assert high == pytest.approx(0.0, abs=1e-9)         # 高波 regime 空仓 (risk-off)
    assert expo.iloc[-1] == 0.0
    assert set(np.unique(w.values)) <= {0.0, 1.0 / data.n_assets}


def test_atr_breakout_flat_then_breakout_holds():
    flat = np.full(60, 100.0)
    up = np.linspace(100.5, 140.0, 25)
    data = _series_market(np.concatenate([flat, up]))
    w = AtrBreakout().generate_weights(data)
    pos = w["A0"]
    assert (pos.iloc[5:55].values == 0.0).all()         # 平坦段不触发
    assert pos.iloc[70] > 0.0                           # 突破后已建仓
    assert (pos.iloc[-10:].values > 0.0).all()          # 趋势中持续持有（状态机不抖出）
    assert pos.iloc[-1] == pytest.approx(1.0, abs=1e-9)  # 单资产等预算 -> 满仓


def test_atr_breakout_exits_when_price_falls_back_to_ma():
    flat = np.full(60, 100.0)
    up = np.linspace(100.5, 140.0, 25)
    down = np.linspace(139.0, 100.0, 20)
    data = _series_market(np.concatenate([flat, up, down]))
    w = AtrBreakout().generate_weights(data)
    pos = w["A0"]
    assert pos.iloc[60:85].max() > 0.0                  # 突破段确实持有过
    assert pos.iloc[-1] == 0.0                          # 跌回均线后离场


def test_atr_breakout_wider_channel_in_high_vol():
    """同样的涨幅，高波动背景下 ATR 通道更宽 -> 触发更晚/更少（波动自适应）。"""
    n = 120
    t = np.arange(n)
    quiet = 100.0 * np.exp(0.001 * t + 0.0005 * np.sin(t / 3.0))
    choppy = quiet * np.exp(np.where(t % 2 == 0, 0.02, -0.02))
    a_q = _close_atr(pd.DataFrame({"A": quiet}), 14)
    a_c = _close_atr(pd.DataFrame({"A": choppy}), 14)
    assert (a_c["A"].iloc[-40:].mean() / a_q["A"].iloc[-40:].mean()) > 5.0


def test_vol_scaled_momentum_prefers_low_vol_at_equal_momentum():
    n = 120
    a, b = 0.03, 1.0 - np.exp(0.002) / 1.03            # (1+a)(1-b) = exp(0.002)
    hi_r = np.tile([a, -b], n // 2)                     # 高波动、交替涨跌
    px = pd.DataFrame({
        "LOW": 100.0 * np.exp(np.cumsum(np.full(n, 0.001))),   # 低波动、稳步上行
        "HIGH": 100.0 * np.cumprod(1.0 + hi_r),
    }, index=pd.bdate_range("2020-01-01", periods=n))
    data = MarketData(prices=px, periods_per_year=252)
    mom = px.shift(5).pct_change(60)
    assert mom["LOW"].iloc[-1] == pytest.approx(mom["HIGH"].iloc[-1], rel=1e-6)  # 动量相同
    w = VolScaledMomentum().generate_weights(data)
    last = w.iloc[-1]
    assert last["LOW"] > 0.0 and last["HIGH"] > 0.0
    assert last["LOW"] > 3.0 * last["HIGH"]             # 波动倒数加权 -> 低波多配
    assert last.sum() == pytest.approx(1.0, abs=1e-9)   # 两个都为正动量 -> 满仓


def test_vol_scaled_momentum_flat_when_all_negative():
    n = 100
    px = pd.DataFrame({"A%d" % i: 100.0 * np.exp(np.cumsum(np.full(n, -0.004)))
                       for i in range(3)},
                      index=pd.bdate_range("2020-01-01", periods=n))
    w = VolScaledMomentum().generate_weights(MarketData(prices=px, periods_per_year=252))
    assert (w.iloc[-20:].values == 0.0).all()           # 全市场负动量 -> 空仓


def test_vol_scaled_momentum_breadth_scales_exposure():
    n = 100
    up = 100.0 * np.exp(np.cumsum(np.full(n, 0.002)))
    down = 100.0 * np.exp(np.cumsum(np.full(n, -0.002)))
    px = pd.DataFrame({"U0": up, "U1": up * 1.01, "D0": down, "D1": down * 0.99},
                      index=pd.bdate_range("2020-01-01", periods=n))
    w = VolScaledMomentum().generate_weights(MarketData(prices=px, periods_per_year=252))
    last = w.iloc[-1]
    assert last["D0"] == 0.0 and last["D1"] == 0.0
    assert last.sum() == pytest.approx(0.5, abs=1e-9)   # 2/4 资产正动量 -> 半仓
