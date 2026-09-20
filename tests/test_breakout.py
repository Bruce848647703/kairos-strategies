"""breakout 渠道测试：形状/边界/确定性/无未来函数 + 行为断言（突破持有、横盘空仓、假突破拒绝、时间止损）。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.breakout import (
    ChannelAtrBreakout,
    KeltnerBreakout,
    RangeBreakout,
    VolatilityBreakout,
    _close_atr,
    _mean_abs_move,
)

EPS = 1e-9
ALL = [RangeBreakout, KeltnerBreakout, VolatilityBreakout, ChannelAtrBreakout]
NAMES = {"range_breakout", "keltner_breakout", "volatility_breakout", "channel_atr_breakout"}


# ---------------------------------------------------------------- 数据构造
def _series_market(values, name: str = "A0") -> MarketData:
    """单资产确定性行情。"""
    px = pd.DataFrame({name: np.asarray(values, dtype=float)},
                      index=pd.bdate_range("2020-01-01", periods=len(values)))
    return MarketData(prices=px, periods_per_year=252)


def _flat_market(n: int = 250) -> MarketData:
    """纯横盘（无突破）。"""
    return _series_market(np.full(n, 100.0))


def _zigzag_market(n: int = 250) -> MarketData:
    """箱体内等幅锯齿震荡：峰值恒定，永不创出「超过前高」的新高（无突破）。"""
    t = np.arange(n)
    return _series_market(np.where(t % 2 == 0, 100.0 * 1.003, 100.0 * 0.997))


def _breakout_market(n_flat: int = 80, n_up: int = 40) -> MarketData:
    """长期横盘后向上突破：前 n_flat 天恒定 100，其后每天 +0.75 线性上行至 130。"""
    flat = np.full(n_flat, 100.0)
    up = np.linspace(100.75, 130.0, n_up)          # 步长 0.75
    return _series_market(np.concatenate([flat, up]))


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
def test_equal_budget_values(cls):
    """0/1 状态机 × 等预算：权重只能是 0 或 1/N。"""
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=2)
    w = cls().generate_weights(data)
    ok = (w.values == 0.0) | (np.abs(w.values * data.n_assets - 1.0) < 1e-9)
    assert ok.all()


@pytest.mark.parametrize("cls", ALL)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=3)
    s = cls()
    pd.testing.assert_frame_equal(s.generate_weights(data), s.generate_weights(data))


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                       # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "breakout"
    assert m["long_only"] is True
    assert m["universe"] == "timing"
    assert m["name"] in NAMES
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique_within_channel():
    got = [cls().name for cls in ALL]
    assert len(set(got)) == len(got) == 4
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


# ---------------------------------------------------------------- 行为断言
def test_range_breakout_flat_then_breakout_holds():
    """长期横盘后向上突破：突破前空仓、突破日起持有并一路拿到结尾。"""
    data = _breakout_market()
    w = RangeBreakout().generate_weights(data)
    pos = w["A0"]
    assert (pos.iloc[:80].values == 0.0).all()        # 横盘段不触发
    assert pos.iloc[80] > 0.0                         # 突破当日入场
    assert (pos.iloc[-10:].values > 0.0).all()        # 趋势中持续持有（中轴未跌破）
    assert pos.iloc[-1] == pytest.approx(1.0, abs=EPS)  # 单资产等预算 -> 满仓


def test_channel_atr_breakout_flat_then_breakout_holds():
    """真突破（幅度远超 k×ATR）：突破前空仓、突破后持有到结尾。"""
    data = _breakout_market()
    w = ChannelAtrBreakout().generate_weights(data)
    pos = w["A0"]
    assert (pos.iloc[:80].values == 0.0).all()
    assert pos.iloc[80] > 0.0
    assert (pos.iloc[-10:].values > 0.0).all()
    assert pos.iloc[-1] == pytest.approx(1.0, abs=EPS)


@pytest.mark.parametrize("cls", ALL)
def test_flat_market_stays_out(cls):
    """纯横盘（零波动、无任何突破）：四个策略全程空仓。"""
    w = cls().generate_weights(_flat_market())
    assert (w.values == 0.0).all()


@pytest.mark.parametrize("cls", ALL)
def test_zigzag_range_market_stays_out(cls):
    """箱体锯齿震荡（峰值恒定、从不超越前高）：全程空仓。"""
    w = cls().generate_weights(_zigzag_market())
    assert (w.values == 0.0).all()


def test_channel_atr_breakout_rejects_false_breakout():
    """假突破过滤：刺穿上沿的幅度 < k×ATR（下限）时 channel_atr_breakout 不入场。"""
    poke = np.concatenate([np.full(80, 100.0), [100.05], np.full(39, 100.0)])
    w = ChannelAtrBreakout().generate_weights(_series_market(poke))
    assert (w.values == 0.0).all()                    # 0.05 <= k×ATR_floor = 0.5×0.2 = 0.1 -> 拒绝


def test_channel_atr_breakout_accepts_confirmed_breakout():
    """同背景真突破：刺穿幅度 > k×ATR（下限）时确认入场（与假突破用例成对）。"""
    poke = np.concatenate([np.full(80, 100.0), [100.5], np.full(39, 100.0)])
    w = ChannelAtrBreakout().generate_weights(_series_market(poke))
    pos = w["A0"]
    assert pos.iloc[80] > 0.0                         # 0.5 > 0.1 -> 确认做多
    assert (pos.iloc[:80].values == 0.0).all()


def test_keltner_breakout_enters_on_upper_band_and_exits_on_crash():
    """肯特纳：横盘后突破上轨入场；深跌破位 EMA 中轨（带缓冲）后离场。"""
    up = np.linspace(100.75, 130.0, 40)
    crash = np.linspace(128.0, 85.0, 20)
    data = _series_market(np.concatenate([np.full(80, 100.0), up, crash]))
    pos = KeltnerBreakout().generate_weights(data)["A0"]
    assert (pos.iloc[:80].values == 0.0).all()        # 横盘段空仓
    assert pos.iloc[80] > 0.0                         # 突破 EMA+2×ATR 上轨入场
    assert (pos.iloc[80:110].values > 0.0).all()      # 上行趋势中持有
    assert pos.iloc[-1] == 0.0                        # 崩盘跌破中轨后已离场


def test_volatility_breakout_enters_on_trigger_price():
    """波动率突破：收盘站上「前收 + k×近期波幅」触发价当日入场。"""
    data = _breakout_market()
    pos = VolatilityBreakout().generate_weights(data)["A0"]
    assert (pos.iloc[:80].values == 0.0).all()
    assert pos.iloc[80] > 0.0                         # 0.75 > 1.2×波幅下限 0.24 -> 触发
    assert (pos.iloc[80:88].values > 0.0).all()


def test_volatility_breakout_time_stop_forces_exit():
    """时间止损：涨速回落到常态波幅以内后，持有满 max_hold 期被强制离场。"""
    data = _breakout_market()
    pos = VolatilityBreakout().generate_weights(data)["A0"]
    assert pos.iloc[90] > 0.0                         # 尚在持有窗口内
    assert (pos.iloc[-10:].values == 0.0).all()       # 已被时间止损清仓且不再重入


def test_volatility_breakout_pullback_exit():
    """回落离场：入场次日出现「跌破 前收 - k×波幅」的大阴线，立即清仓。"""
    vals = np.concatenate([np.full(80, 100.0), [103.0, 100.5], np.full(38, 100.5)])
    pos = VolatilityBreakout().generate_weights(_series_market(vals))["A0"]
    assert pos.iloc[80] > 0.0                         # +3 大阳线站上触发价 -> 入场
    assert pos.iloc[81] == 0.0                        # -2.5 跌破回落触发价 -> 次日离场
    assert (pos.iloc[82:].values == 0.0).all()        # 此后无新突破，保持空仓


# ---------------------------------------------------------------- 近似工具
def test_close_atr_matches_mean_abs_move_scale():
    """收盘近似 ATR：常数波动序列的 ATR 收敛到 |diff| 常数本身（Wilder 平滑的不动点）。"""
    vals = np.concatenate([[100.0], 100.0 + np.cumsum(np.full(99, 0.6))])
    px = pd.DataFrame({"A": vals}, index=pd.bdate_range("2020-01-01", periods=100))
    a = _close_atr(px, 14)["A"]
    assert a.iloc[-1] == pytest.approx(0.6, rel=1e-6)
    m = _mean_abs_move(px, 10)["A"]
    assert m.iloc[-1] == pytest.approx(0.6, rel=1e-9)
