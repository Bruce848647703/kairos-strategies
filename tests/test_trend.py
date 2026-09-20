"""trend 渠道测试：形状/边界/确定性/无未来函数 + 行为断言（上涨持有、低波多配）。

注：本测试**直接 import 本渠道模块**，不调用 ks.discover()，以免与其它并行渠道相互干扰。
"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.trend import (
    AdxTrend,
    DualThrust,
    MaRibbon,
    TsMomVolScaled,
    TurtleAtr,
    _close_atr,
    _state_machine,
)

EPS = 1e-9
ALL = [TsMomVolScaled, MaRibbon, AdxTrend, DualThrust, TurtleAtr]
NAMES = {"tsmom_volscaled", "ma_ribbon", "adx_trend", "dual_thrust", "turtle_atr"}


# ---------------------------------------------------------------- 数据构造
def _series_market(values, name: str = "A0") -> MarketData:
    px = pd.DataFrame({name: np.asarray(values, dtype=float)},
                      index=pd.bdate_range("2020-01-01", periods=len(values)))
    return MarketData(prices=px, periods_per_year=252)


def _uptrend_market(n: int = 250, drift: float = 0.002) -> MarketData:
    """单边稳步上涨（低波动、强趋势）的单资产行情。"""
    return _series_market(100.0 * np.exp(np.cumsum(np.full(n, drift))))


def _two_vol_market(n: int = 220) -> MarketData:
    """LOW(稳步上涨,低波动) 与 HIGH(同样净涨幅,交替涨跌,高波动) 的两资产行情。

    HIGH 每两日 (1+a)(1-b)=exp(2*drift)，使其 N 期净收益与 LOW 一致而波动远大，
    用于验证「同等趋势下低波动 -> 仓位更大」。
    """
    drift = 0.002
    a = 0.04
    b = 1.0 - np.exp(2.0 * drift) / (1.0 + a)
    hi_r = np.tile([a, -b], n // 2)
    px = pd.DataFrame({
        "LOW": 100.0 * np.exp(np.cumsum(np.full(n, drift))),
        "HIGH": 100.0 * np.cumprod(1.0 + hi_r),
    }, index=pd.bdate_range("2020-01-01", periods=n))
    return MarketData(prices=px, periods_per_year=252)


# ---------------------------------------------------------------- 通用契约
@pytest.mark.parametrize("cls", ALL)
def test_no_arg_constructible(cls):
    s = cls()                              # 必须无参可构造
    assert s.channel == "trend"
    assert s.long_only is True


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
    assert (w.values >= 0.0).all()                          # long_only 权重 ≥ 0
    assert np.abs(w.values).sum(axis=1).max() <= 1.0 + EPS  # 每行和 ≤ 1
    assert np.abs(w.values).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=3)
    s = cls()
    pd.testing.assert_frame_equal(s.generate_weights(data), s.generate_weights(data))


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    m = cls().meta()
    assert m["channel"] == "trend"
    assert m["long_only"] is True
    assert m["name"] in NAMES
    assert m["universe"] in {"timing", "cross_section", "overlay"}
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique():
    got = [cls().name for cls in ALL]
    assert len(set(got)) == len(got) == 5
    assert set(got) == NAMES


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


# ---------------------------------------------------------------- 辅助函数
def test_close_atr_is_mean_abs_move():
    """收盘价近似 ATR = |Δclose| 的 Wilder 平滑：稳步上涨序列 ATR≈每日绝对变动。"""
    n = 120
    px = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(np.full(n, 0.003)))})
    a = _close_atr(px, 14)["A"]
    daily_move = (px["A"].diff().abs()).iloc[-1]
    assert a.iloc[-1] == pytest.approx(daily_move, rel=0.12)   # Wilder 平滑略滞后于上升的 |Δ|
    assert a.iloc[:13].isna().all()                         # min_periods 之前为 NaN


def test_state_machine_holds_until_exit():
    idx = pd.bdate_range("2020-01-01", periods=6)
    enter = pd.DataFrame({"A": [False, True, False, False, False, False]}, index=idx)
    exit_ = pd.DataFrame({"A": [False, False, False, False, True, False]}, index=idx)
    out = _state_machine(enter, exit_)["A"].tolist()
    assert out == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]           # enter 后持有，exit 后清 0


# ---------------------------------------------------------------- 行为断言
# (1) 单边上涨 -> tsmom_volscaled / ma_ribbon / turtle_atr 持有(权重>0)
@pytest.mark.parametrize("cls", [TsMomVolScaled, MaRibbon, TurtleAtr])
def test_uptrend_holds(cls):
    data = _uptrend_market(n=250, drift=0.002)
    w = cls().generate_weights(data)
    pos = w["A0"]
    assert (pos.iloc[-30:].values > 0.0).all()              # 上涨趋势中持续持有
    assert pos.iloc[-1] > 0.0


def test_ma_ribbon_perfect_alignment_full_score():
    """单边上涨 -> 均线完美多头排列 -> 排列分数=1 -> 单资产满仓。"""
    data = _uptrend_market(n=200, drift=0.003)
    w = MaRibbon().generate_weights(data)
    assert w["A0"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_ma_ribbon_flat_no_alignment():
    """完全平坦 -> 各均线相等(非严格大于) -> 排列分数=0 -> 不建仓。"""
    data = _series_market(np.full(160, 100.0))
    w = MaRibbon().generate_weights(data)
    assert (w["A0"].iloc[-20:].values == 0.0).all()


# (2) 高波动 vs 低波动 -> tsmom_volscaled / turtle_atr 在低波动时仓位更大
@pytest.mark.parametrize("cls", [TsMomVolScaled, TurtleAtr])
def test_low_vol_gets_larger_position(cls):
    data = _two_vol_market(n=220)
    w = cls().generate_weights(data)
    tail = w.iloc[-40:]
    low, high = tail["LOW"].mean(), tail["HIGH"].mean()
    assert low > 0.0 and high >= 0.0
    assert low > high                                       # 同等趋势下低波动 -> 仓位更大
    assert low > 1.5 * high                                 # 差距显著


def test_tsmom_flat_when_all_negative():
    """全市场单边下跌 -> 动量为负 -> 空仓。"""
    n = 120
    px = pd.DataFrame({"A%d" % i: 100.0 * np.exp(np.cumsum(np.full(n, -0.004)))
                       for i in range(3)},
                      index=pd.bdate_range("2020-01-01", periods=n))
    w = TsMomVolScaled().generate_weights(MarketData(prices=px, periods_per_year=252))
    assert (w.iloc[-20:].values == 0.0).all()


def test_adx_holds_in_strong_uptrend_flat_in_range():
    data_up = _uptrend_market(n=200, drift=0.003)
    w_up = AdxTrend().generate_weights(data_up)
    assert w_up["A0"].iloc[-1] > 0.0                        # 强趋势 -> ADX 高 -> 持有
    flat = 100.0 + np.sin(np.arange(200) / 5.0)             # 无方向震荡
    w_flat = AdxTrend().generate_weights(_series_market(flat))
    assert w_flat["A0"].iloc[-40:].mean() < w_up["A0"].iloc[-1]


def test_dual_thrust_flat_then_breakout_holds():
    flat = np.full(80, 100.0)
    up = np.linspace(100.5, 150.0, 40)
    data = _series_market(np.concatenate([flat, up]))
    pos = DualThrust().generate_weights(data)["A0"]
    assert pos.iloc[10:78].max() == 0.0                     # 平坦段不触发
    assert pos.iloc[-1] > 0.0                               # 突破后建仓并持有
    assert (pos.iloc[-15:].values > 0.0).all()


def test_dual_thrust_exits_on_breakdown():
    flat = np.full(80, 100.0)
    up = np.linspace(100.5, 150.0, 30)
    down = np.linspace(149.0, 95.0, 30)
    data = _series_market(np.concatenate([flat, up, down]))
    pos = DualThrust().generate_weights(data)["A0"]
    assert pos.iloc[80:108].max() > 0.0                     # 上涨段确实持有过
    assert pos.iloc[-1] == 0.0                              # 跌破下轨后离场


def test_turtle_atr_exits_on_breakdown():
    up = np.linspace(100.0, 160.0, 120)
    down = np.linspace(159.0, 110.0, 40)
    data = _series_market(np.concatenate([up, down]))
    pos = TurtleAtr().generate_weights(data)["A0"]
    assert pos.iloc[60:118].max() > 0.0                     # 突破段持有
    assert pos.iloc[-1] == 0.0                              # 跌破退出通道离场
