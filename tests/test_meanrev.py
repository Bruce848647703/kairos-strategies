"""meanrev 渠道测试。

覆盖：形状/列对齐/有限值、每行绝对值和 <= 1+eps、值域 [-1,1]、确定性、
可多空（权重可正可负）、等预算 1/N、行为断言（深跌做多 / 急涨做空 / 配对两腿反向且中性）、
滞后带降换手、防未来函数（前缀不变性）、元信息与无参构造、registry 自动发现。
"""
import numpy as np
import pandas as pd
import pytest

import kairos_strategies as ks
from kairos_strategies import MarketData
from kairos_strategies import indicators as ind
from kairos_strategies.channels.meanrev import (
    BollingerReversionStrategy,
    MaDeviationStrategy,
    PairsSpreadStrategy,
    ZscoreReversionStrategy,
)

STRATEGIES = [ZscoreReversionStrategy, BollingerReversionStrategy,
              MaDeviationStrategy, PairsSpreadStrategy]
TIMING = [ZscoreReversionStrategy, BollingerReversionStrategy, MaDeviationStrategy]
EPS = 1e-9
TAIL = 5
N_DAYS = 65            # 平稳段 60 + 末段位移 5
WINDOW = 20            # 择时策略默认窗口


def _osc(n, phase=0.0):
    """带轻微波动的平稳价格段（避免滚动标准差为 0）。"""
    return 100.0 * (1.0 + 0.01 * np.sin(np.arange(n) * 1.1 + phase))


def _path(kind, n=N_DAYS):
    """kind='down' 末段急跌至 80（远低于滚动均值）；'up' 末段急涨至 120（远高于）。"""
    head = _osc(n - TAIL)
    tail = np.linspace(100.0, 80.0, TAIL) if kind == "down" else np.linspace(100.0, 120.0, TAIL)
    return np.concatenate([head, tail])


def _two_leg_data(kind):
    """A0 = 位移腿，A1 = 平稳腿；构成 pairs_spread 的价差冲击。"""
    idx = pd.bdate_range("2021-01-01", periods=N_DAYS)
    prices = pd.DataFrame({"A0": _path(kind), "A1": _osc(N_DAYS, phase=0.7)}, index=idx)
    return MarketData(prices=prices)


@pytest.fixture(scope="module")
def synth():
    return ks.make_synthetic_universe(n_assets=6, n_days=400, seed=1)


# ---------------------------------------------------------------- 契约与数值健全

def test_shape_columns_index_alignment(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == (len(synth.dates), len(synth.symbols))
        assert list(w.columns) == synth.symbols
        assert w.index.equals(synth.dates)


def test_all_values_finite(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert np.isfinite(w.values).all(), cls.name


def test_row_abs_sum_le_one(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name


def test_weights_within_unit_range(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() >= -1.0 - EPS
        assert w.values.max() <= 1.0 + EPS


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)


def test_long_short_both_signs_appear(synth):
    """可多空策略必须真的会产生空头（负权重）与多头（正权重）。"""
    for cls in STRATEGIES:
        s = cls()
        assert s.long_only is False
        w = s.generate_weights(synth)
        assert w.values.min() < 0.0, f"{cls.name} 从未做空"
        assert w.values.max() > 0.0, f"{cls.name} 从未做多"


def test_timing_uses_equal_budget_per_asset(synth):
    """择时策略非零权重恒为 ±1/N（等预算分配）。"""
    unit = 1.0 / synth.n_assets
    for cls in TIMING:
        w = cls().generate_weights(synth)
        nz = w.values[w.values != 0.0]
        assert nz.size > 0
        assert np.allclose(np.abs(nz), unit, atol=EPS), cls.name


# ------------------------------------------------------------------ 行为断言

def test_zscore_reversion_longs_deep_dip_and_shorts_spike():
    s = ZscoreReversionStrategy()
    w_down = s.generate_weights(_two_leg_data("down"))
    w_up = s.generate_weights(_two_leg_data("up"))
    assert w_down["A0"].iloc[-1] > 0.0        # 价格远低于滚动均值 -> 做多
    assert w_up["A0"].iloc[-1] < 0.0          # 价格远高于滚动均值 -> 做空
    assert w_down["A0"].iloc[:WINDOW - 1].abs().sum() == 0.0   # 窗口未就绪前不交易


def test_ma_deviation_longs_below_ma_and_shorts_above_ma():
    s = MaDeviationStrategy()
    down, up = _two_leg_data("down"), _two_leg_data("up")
    w_down, w_up = s.generate_weights(down), s.generate_weights(up)
    assert w_down["A0"].iloc[-1] > 0.0
    assert w_up["A0"].iloc[-1] < 0.0
    # 与偏离方向一致性：末段偏离为负 -> 正权重；偏离为正 -> 负权重
    dev_dn = (down.prices["A0"] / ind.sma(down.prices["A0"], WINDOW) - 1.0).iloc[-1]
    dev_up = (up.prices["A0"] / ind.sma(up.prices["A0"], WINDOW) - 1.0).iloc[-1]
    assert dev_dn < -s.params["entry_dev"] and w_down["A0"].iloc[-1] > 0.0
    assert dev_up > s.params["entry_dev"] and w_up["A0"].iloc[-1] < 0.0


def test_bollinger_reversion_is_opposite_of_breakout():
    s = BollingerReversionStrategy()
    down, up = _two_leg_data("down"), _two_leg_data("up")
    w_down, w_up = s.generate_weights(down), s.generate_weights(up)
    assert w_down["A0"].iloc[-1] > 0.0        # 跌破下轨 -> 做多（而非追突破）
    assert w_up["A0"].iloc[-1] < 0.0          # 升破上轨 -> 做空
    mid, upper, lower = ind.bollinger(down.prices, s.params["window"], s.params["num_std"])
    assert down.prices["A0"].iloc[-1] < lower["A0"].iloc[-1]   # 确实在下轨之外


def test_bollinger_reversion_exits_near_mid_band():
    """深跌做多后价格回到中轨附近 -> 平仓（滞后带的退出侧）。"""
    n = 90
    px = np.concatenate([_path("down"), np.full(n - N_DAYS, 80.0) *
                         (1.0 + 0.004 * np.sin(np.arange(n - N_DAYS) * 0.9))])
    data = MarketData(prices=pd.DataFrame({"A0": px, "A1": _osc(n)},
                                          index=pd.bdate_range("2021-01-01", periods=n)))
    w = BollingerReversionStrategy().generate_weights(data)
    assert w["A0"].iloc[N_DAYS] > 0.0         # 跌破后建仓
    assert w["A0"].iloc[-1] == 0.0            # 均值收敛后已平仓


def test_pairs_spread_legs_opposite_and_market_neutral():
    s = PairsSpreadStrategy()
    w = s.generate_weights(_two_leg_data("down"))
    last = w.iloc[-1]
    assert last["A0"] > 0.0 and last["A1"] < 0.0             # A0 相对被低估 -> 多 A0 空 A1
    assert np.isclose(last["A0"], -last["A1"], atol=EPS)      # 两腿等权反向
    active = w.abs().sum(axis=1) > 0
    assert np.allclose(w[active].sum(axis=1), 0.0, atol=EPS)   # 每行权重和 ≈ 0（市场中性）
    assert w.abs().sum(axis=1).max() <= 1.0 + EPS


def test_pairs_spread_reverses_when_leg_overvalued():
    s = PairsSpreadStrategy()
    w = s.generate_weights(_two_leg_data("up"))                # A0 急涨 -> 价差被高估
    assert w["A0"].iloc[-1] < 0.0 and w["A1"].iloc[-1] > 0.0


def test_pairs_spread_flat_before_window_ready_and_untouched_legs():
    s = PairsSpreadStrategy()
    w = s.generate_weights(_two_leg_data("down"))
    assert (w.iloc[:s.params["window"] - 1].values == 0.0).all()
    # 只有配对两腿有权重，其余资产恒为 0
    data = ks.make_synthetic_universe(n_assets=6, n_days=200, seed=7)
    w6 = s.generate_weights(data)
    others = [c for c in data.symbols if c not in (s.params["leg_a"], s.params["leg_b"])]
    assert (w6[others].values == 0.0).all()
    assert (w6[[s.params["leg_a"], s.params["leg_b"]]].values != 0.0).any()


def test_pairs_spread_fallback_legs_and_single_asset():
    s = PairsSpreadStrategy()
    base = ks.make_synthetic_universe(n_assets=3, n_days=200, seed=3)
    odd = MarketData(prices=base.prices.rename(columns={"A0": "X", "A1": "Y", "A2": "Z"}))
    w = s.generate_weights(odd)                                 # 代码非 A0/A1 -> 退回前两列
    assert (w["Z"].values == 0.0).all()
    assert (w[["X", "Y"]].values != 0.0).any()
    w1 = s.generate_weights(MarketData(prices=base.prices[["A0"]]))
    assert (w1.values == 0.0).all()                             # 单资产无法配对 -> 空仓


def test_hysteresis_reduces_turnover_vs_naive_threshold(synth):
    """滞后带状态机的换手应显著低于「无状态直接阈值」写法。"""
    s = ZscoreReversionStrategy()
    w = s.generate_weights(synth)
    z = ind.rolling_zscore(np.log(synth.prices.astype("float64")), s.params["window"])
    naive = pd.DataFrame(0.0, index=synth.dates, columns=synth.symbols)
    naive[z < -s.params["entry_z"]] = 1.0
    naive[z > s.params["entry_z"]] = -1.0
    naive = naive / synth.n_assets
    turn_state = np.abs(np.diff(w.values, axis=0)).sum()
    turn_naive = np.abs(np.diff(naive.values, axis=0)).sum()
    assert turn_state < turn_naive


# ------------------------------------------------------------ 防未来函数 / 发现

def test_no_lookahead_prefix_invariance(synth):
    """只用历史：截断样本重算，前缀权重必须与全样本完全一致。"""
    m = 200
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.allclose(full, part, atol=0.0, rtol=0.0), cls.name


def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                       # 无参可构造
        meta = s.meta()
        assert s.channel == "meanrev"
        assert s.name and s.name == s.name.strip().lower()
        assert s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 10, (s.name, key)
        assert isinstance(meta["params"], dict)
        assert s.universe in ("timing", "cross_section")
    for cls in TIMING:
        assert cls.universe == "timing"
    assert PairsSpreadStrategy.universe == "cross_section"
    assert PairsSpreadStrategy.long_only is False


def test_discover_finds_all_meanrev_strategies():
    found = {s.name: s for s in ks.discover() if s.channel == "meanrev"}
    assert set(found) == {cls.name for cls in STRATEGIES}
    data = ks.make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    for s in found.values():
        assert s.generate_weights(data).shape == (400, 6)
