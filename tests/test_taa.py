"""taa 渠道测试：契约（形状/边界/确定性/无未来函数）+ 四类战术配置的行为断言。

不调用 ``ks.discover()``：直接 import 本渠道模块与策略类。
"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.taa import (
    STRATEGY_NAMES,
    DualMomentumTaaStrategy,
    Mom12mTaaStrategy,
    TrendRegimeTaaStrategy,
    VolTargetTaaStrategy,
    _cash_return,
    _drift_hold,
    _ewma_vol,
    _inverse_vol_sleeve,
    _market_index,
    _n_slots,
    _sma,
    _trailing_mom,
)

EPS = 1e-9
ALL = [DualMomentumTaaStrategy, TrendRegimeTaaStrategy, VolTargetTaaStrategy,
       Mom12mTaaStrategy]
NAMES = {"dual_momentum_taa", "trend_regime_taa", "vol_target_taa", "mom_12m_taa"}
# 仓库既有策略名（硬编码，避免调用 discover()）：本渠道不得与之重名
EXISTING_NAMES = frozenset({
    "adx_trend", "atr_breakout", "basket_neutral", "bollinger_breakout",
    "bollinger_reversion", "carry_proxy", "channel_atr_breakout", "coint_pairs",
    "crypto_momentum_247", "dca", "donchian_turtle", "dual_momentum", "dual_thrust",
    "eof_stat_arb", "equal_weight_buy_hold", "equal_weight_rebal", "grid_trading",
    "high_52w", "idio_momentum", "inverse_vol", "keltner_breakout", "low_volatility",
    "ma_deviation", "ma_ribbon", "macd_trend", "max_diversification",
    "min_variance_alloc", "month_of_year", "pairs_spread", "range_breakout",
    "risk_parity_alloc", "rsi_reversion", "short_term_reversal", "sma_cross",
    "trend_quality", "ts_momentum", "tsmom_volscaled", "turn_of_month",
    "turtle_atr", "vol_regime_filter", "vol_scaled_momentum", "vol_target",
    "volatility_breakout", "weekday_effect", "xs_momentum", "xs_zscore_reversion",
    "zscore_reversion",
})


# ---------------------------------------------------------------- 数据构造
def _bull_bear_market(n_up: int = 300, n_down: int = 150, up: float = 0.002,
                      down: float = -0.006, n_assets: int = 4):
    """先单边上涨（市场指数远高于长期均线）后单边下跌（跌破均线）的确定性行情。"""
    n = n_up + n_down
    r = np.concatenate([np.full(n_up, up), np.full(n_down, down)])
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        cols["A%d" % i] = 100.0 * np.cumprod(1.0 + r * (1.0 + 0.1 * i))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_up


def _vol_regime_market(n_low: int = 260, n_high: int = 90, low_sigma: float = 0.0015,
                       high_sigma: float = 0.045, drift: float = 0.0004, seed: int = 11,
                       n_assets: int = 4):
    """前 n_low 天低波动、后 n_high 天高波动的合成行情（固定 seed，确定性）。"""
    rng = np.random.default_rng(seed)
    n = n_low + n_high
    sigma = np.concatenate([np.full(n_low, low_sigma), np.full(n_high, high_sigma)])
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        r = drift * (1.0 + 0.1 * i) + rng.standard_normal(n) * sigma
        cols["A%d" % i] = 100.0 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_low


def _dispersion_market(n: int = 420, n_strong: int = 1, up: float = 0.003,
                       down: float = -0.002, n_assets: int = 6):
    """一强（或少数强）多弱：前 n_strong 个资产稳定上行，其余稳定下行。"""
    idx = pd.bdate_range("2017-01-02", periods=n)
    cols = {}
    for i in range(n_assets):
        r = (up if i < n_strong else down) * (1.0 + 0.05 * i)
        cols["A%d" % i] = 100.0 * np.cumprod(1.0 + np.full(n, r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252)


def _mild_market(n: int = 300, mild: float = 0.00003, down: float = -0.0005,
                 n_assets: int = 6):
    """最强资产也只是微涨（跑不赢现金利率），其余下跌——用于验证现金门槛。"""
    idx = pd.bdate_range("2017-01-02", periods=n)
    cols = {}
    for i in range(n_assets):
        r = mild if i == 0 else down * (1.0 + 0.05 * i)
        cols["A%d" % i] = 100.0 * np.cumprod(1.0 + np.full(n, r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252)


def _exposure(w: pd.DataFrame) -> pd.Series:
    """总敞口 = 每行权重和；``1 - 敞口`` 即现金比例。"""
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
    assert np.isfinite(w.to_numpy()).all()


@pytest.mark.parametrize("cls", ALL)
def test_long_only_and_row_sum_cap(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    w = cls().generate_weights(data)
    assert (w.to_numpy() >= 0.0).all()                       # long_only
    assert np.abs(w.to_numpy()).sum(axis=1).max() <= 1.0 + EPS   # 每行和 ≤ 1（不加杠杆）
    assert np.abs(w.to_numpy()).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic_across_calls_and_instances(cls):
    data = make_synthetic_universe(n_assets=5, n_days=320, seed=3)
    s = cls()
    a, b = s.generate_weights(data), s.generate_weights(data)
    pd.testing.assert_frame_equal(a, b)                      # 同实例两次一致
    pd.testing.assert_frame_equal(a, cls().generate_weights(data))  # 新实例一致


@pytest.mark.parametrize("cls", ALL)
def test_no_lookahead_future_prices_do_not_change_history(cls):
    """改动 t 之后的价格，t 及之前的权重必须完全不变（自证无未来函数）。"""
    data = make_synthetic_universe(n_assets=5, n_days=320, seed=5)
    t = 200
    base = cls().generate_weights(data)
    p2 = data.prices.copy()
    ramp = np.linspace(1.0, 2.5, len(p2) - t - 1)[:, None]
    p2.iloc[t + 1:] = p2.iloc[t + 1:].to_numpy() * ramp
    mutated = cls().generate_weights(MarketData(prices=p2, periods_per_year=252))
    pd.testing.assert_frame_equal(base.iloc[:t + 1], mutated.iloc[:t + 1])


@pytest.mark.parametrize("cls", ALL)
def test_warmup_is_all_cash(cls):
    """首期无任何历史统计量 → 全部现金（不用未来数据回填预热期）。"""
    data = make_synthetic_universe(n_assets=4, n_days=260, seed=9)
    w = cls().generate_weights(data)
    assert w.iloc[0].abs().sum() == 0.0


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                                                # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "taa"
    assert m["universe"] == "cross_section"
    assert m["long_only"] is True
    assert m["name"] in NAMES
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique_and_no_collision_with_existing_channels():
    got = [cls().name for cls in ALL]
    assert len(got) == len(set(got)) == 4
    assert set(got) == NAMES == set(STRATEGY_NAMES)
    assert not (NAMES & EXISTING_NAMES)                      # 不与既有策略重名


def test_cash_is_the_residual_of_row_sum():
    """现金 = 1 - 行和：任何一行都不允许出现『行和 > 1』（即借钱买风险资产）。"""
    data = make_synthetic_universe(n_assets=8, n_days=600, seed=2)
    for cls in ALL:
        expo = _exposure(cls().generate_weights(data))
        assert expo.max() <= 1.0 + EPS
        assert (expo.to_numpy() >= -EPS).all()
        assert (expo.to_numpy() < 1.0 - 1e-6).any()        # 确有『部分现金』的时段


# ---------------------------------------------------------------- 行为断言
def test_trend_regime_full_exposure_above_ma_and_cash_below():
    """市场在长期均线上方 → 满仓等权；跌破均线 → 总敞口 ≈ 0（全部转现金）。"""
    data, n_up = _bull_bear_market()
    w = TrendRegimeTaaStrategy().generate_weights(data)
    expo = _exposure(w)
    mkt = _market_index(data)
    ma = _sma(mkt, TrendRegimeTaaStrategy.params["ma_window"])
    bull = expo.iloc[n_up - 80:n_up]                          # 上涨尾段：指数远高于均线
    bear = expo.iloc[-80:]                                    # 下跌尾段：指数远低于均线
    assert (mkt.iloc[n_up - 80:n_up] > ma.iloc[n_up - 80:n_up]).all()
    assert (mkt.iloc[-80:] < ma.iloc[-80:]).all()
    assert bull.mean() == pytest.approx(1.0, abs=1e-12)       # risk-on：满仓
    assert bear.max() == pytest.approx(0.0, abs=1e-12)        # risk-off：全部现金
    per = 1.0 / data.n_assets
    on_rows = w.iloc[n_up - 80:n_up].to_numpy()
    assert np.allclose(on_rows, per, atol=1e-12)              # 等权 sleeve
    assert expo.iloc[:199].abs().sum() == 0.0                 # 均线预热期不建仓


def test_trend_regime_state_switch_is_hysteresis_not_daily_flip():
    """滞后带 + 状态机：均线附近不逐日抖动，状态段数远少于天数。"""
    n = 400
    t = np.arange(n)
    wobble = 0.0015 * np.sin(t / 9.0)                         # 围绕趋势反复穿越均线
    r = 0.0004 + wobble
    px = pd.DataFrame({"A%d" % i: 100.0 * np.exp(np.cumsum(r * (1 + 0.02 * i)))
                       for i in range(3)},
                      index=pd.bdate_range("2018-01-01", periods=n))
    s = TrendRegimeTaaStrategy()
    s.params = dict(s.params, ma_window=60)                   # 实例级覆盖，不改类属性
    expo = _exposure(s.generate_weights(MarketData(prices=px, periods_per_year=252)))
    states = np.unique(expo.to_numpy())
    assert set(states) <= {0.0, 1.0}                            # 只有满仓 / 现金两种状态
    flips = int((np.diff(expo.to_numpy()) != 0).sum())
    assert flips < 0.1 * n                                      # 切换次数远少于天数


def test_trend_regime_momentum_sleeve_still_full_when_risk_on():
    """sleeve='momentum' 时，risk-on 仍满仓（行和=1），且权重向强势资产倾斜。"""
    data, n_up = _bull_bear_market(n_assets=4)
    s = TrendRegimeTaaStrategy()
    s.params = dict(s.params, sleeve="momentum")
    w = s.generate_weights(data)
    expo = _exposure(w)
    assert expo.iloc[n_up - 60:n_up].mean() == pytest.approx(1.0, abs=1e-9)
    assert expo.iloc[-60:].max() == pytest.approx(0.0, abs=1e-12)
    last_on = w.iloc[n_up - 1]
    assert last_on.max() > 1.0 / data.n_assets                 # 非等权：集中在强动量资产
    assert (last_on >= 0.0).all()


def test_vol_target_deleverages_in_high_vol_and_is_capped_at_one():
    """低波段满仓（缩放触顶，不加杠杆）；高波段总敞口显著更低（余额为现金）。"""
    data, n_low = _vol_regime_market()
    w = VolTargetTaaStrategy().generate_weights(data)
    expo = _exposure(w)
    low = expo.iloc[80:n_low]
    high = expo.iloc[n_low + 30:]
    assert low.mean() == pytest.approx(1.0, abs=1e-9)          # 低波：目标波动 > 已实现 → 满仓
    assert high.mean() < low.mean()                            # 高波：降仓
    assert high.mean() < 0.5
    assert expo.max() <= 1.0 + EPS
    assert (expo.to_numpy() >= 0.0).all()


def test_vol_target_scale_equals_target_over_realized_sleeve_vol():
    """精确性：总敞口 = clip(目标波动 / sleeve EWMA 已实现波动, 0, 1)，波动未知则空仓。"""
    data, _ = _vol_regime_market()
    p = VolTargetTaaStrategy.params
    sleeve = _inverse_vol_sleeve(data, int(p["sleeve_window"]), float(p["vol_floor"]))
    sleeve_ret = (sleeve.shift(1).fillna(0.0) * data.returns(1)).sum(axis=1)
    rv_raw = _ewma_vol(sleeve_ret, int(p["vol_span"]), data.periods_per_year)
    rv = rv_raw.clip(lower=float(p["vol_floor"]))
    expected = np.clip(p["target_vol"] / rv, p["min_scale"], p["max_scale"])
    expo = _exposure(VolTargetTaaStrategy().generate_weights(data))
    ok = rv_raw.notna()
    assert np.allclose(expo[ok].to_numpy(), expected[ok].to_numpy(), atol=1e-12)
    assert (expo[~ok].to_numpy() == 0.0).all()                 # 波动未知 → 全现金
    assert sleeve.sum(axis=1).iloc[-1] == pytest.approx(1.0, abs=1e-12)  # sleeve 内部归一


def test_vol_target_exposure_monotone_in_vol_level():
    """同一策略：波动越高，总敞口越低（三段阶梯波动的单调性）。"""
    seg = 120
    sigmas = [0.002, 0.012, 0.040]
    n = seg * len(sigmas)
    rng = np.random.default_rng(23)
    sigma = np.concatenate([np.full(seg, s) for s in sigmas])
    idx = pd.bdate_range("2016-01-01", periods=n)
    px = pd.DataFrame({"A%d" % i: 100.0 * np.exp(np.cumsum(0.0003 + rng.standard_normal(n) * sigma))
                       for i in range(3)}, index=idx)
    expo = _exposure(VolTargetTaaStrategy().generate_weights(
        MarketData(prices=px, periods_per_year=252)))
    means = [expo.iloc[i * seg + 40:(i + 1) * seg].mean() for i in range(len(sigmas))]
    assert means[0] > means[1] > means[2]


def test_dual_momentum_holds_only_strong_and_parks_rest_in_cash():
    """一强多弱：只有相对动量最强且绝对动量为正的资产被持有，其余槽位配现金。"""
    data = _dispersion_market(n=420, n_strong=1, n_assets=6)
    s = DualMomentumTaaStrategy()
    w = s.generate_weights(data)
    n_slots = _n_slots(data.n_assets, s.params["top_frac"])
    assert n_slots == 2
    last = w.iloc[-1]
    assert last["A0"] == pytest.approx(1.0 / n_slots, abs=1e-12)   # 强势资产占一个槽位
    assert (last.drop(index="A0").to_numpy() == 0.0).all()          # 弱势资产全为 0
    assert _exposure(w).iloc[-1] == pytest.approx(1.0 / n_slots, abs=1e-12)  # 另一槽 = 现金


def test_dual_momentum_two_strong_assets_fill_all_slots():
    """两个强势资产时槽位填满 → 总敞口 1（无现金），弱势资产仍为 0。"""
    data = _dispersion_market(n=420, n_strong=2, n_assets=6)
    w = DualMomentumTaaStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last["A0"] > 0.0 and last["A1"] > 0.0
    assert (last[["A2", "A3", "A4", "A5"]].to_numpy() == 0.0).all()
    assert last.sum() == pytest.approx(1.0, abs=1e-12)


def test_dual_momentum_cash_hurdle_beats_positive_but_weak_momentum():
    """绝对动量门槛：最强资产虽为正收益但跑不赢现金 → 整个组合转现金（敞口 0）。"""
    data = _mild_market()
    s = DualMomentumTaaStrategy()
    mom = _trailing_mom(data.prices, int(s.params["lookback"]))
    hurdle = _cash_return(int(s.params["lookback"]), float(s.params["cash_annual"]),
                          data.periods_per_year)
    assert mom["A0"].iloc[-1] > 0.0                       # 确实为正（相对最强）
    assert mom["A0"].iloc[-1] < hurdle                    # 但跑不赢现金
    assert (mom.iloc[:, 1:].iloc[-1] < 0.0).all()         # 其余为负
    w = s.generate_weights(data)
    assert (w.iloc[200:].to_numpy() == 0.0).all()         # 全现金
    assert _exposure(w).iloc[-1] == 0.0


def _last_rebal_index(n: int, rebal: int) -> int:
    """最后一个再平衡日的行号（权重在该行被重置为目标权重，无漂移污染）。"""
    return ((n - 1) // max(int(rebal), 1)) * max(int(rebal), 1)


def test_mom12m_holds_top_assets_and_substitutes_cash_for_negative_momentum():
    """12-1 动量：top-N 等权持有，动量为负的槽位用现金替代（敞口 = 合格槽位/n_slots）。"""
    data = _dispersion_market(n=420, n_strong=1, n_assets=6)
    s = Mom12mTaaStrategy()
    w = s.generate_weights(data)
    rebal = int(s.params["rebal"])
    n_slots = _n_slots(data.n_assets, s.params["top_frac"])
    assert n_slots == 2
    t_r = _last_rebal_index(len(w), rebal)
    on_rebal = w.iloc[t_r]
    assert on_rebal["A0"] == pytest.approx(1.0 / n_slots, abs=1e-12)   # 强势资产占一个槽位
    assert (on_rebal.drop(index="A0").to_numpy() == 0.0).all()         # 弱势资产全为 0
    assert _exposure(w).iloc[t_r] == pytest.approx(1.0 / n_slots, abs=1e-12)  # 另一槽 = 现金
    last = w.iloc[-1]
    assert last["A0"] > 0.0 and (last.drop(index="A0").to_numpy() == 0.0).all()
    assert 1.0 / n_slots - EPS <= last.sum() <= 1.0 + EPS   # 漂移后仍不加杠杆
    warmup = int(s.params["lookback"]) + int(s.params["skip"])
    assert (w.iloc[:warmup].to_numpy() == 0.0).all()        # 12-1 窗口未满 → 全现金


def test_mom12m_rebalances_only_on_schedule_and_drifts_in_between():
    """再平衡纪律：非再平衡日的权重 = 上一期权重按净值漂移（买入持有，不产生杠杆）。"""
    data = _dispersion_market(n=420, n_strong=2, n_assets=6)
    s = Mom12mTaaStrategy()
    rebal = int(s.params["rebal"])
    w = s.generate_weights(data)
    rets = np.nan_to_num(data.returns(1).to_numpy(dtype=float))
    checked = 0
    for t in range(300, len(w)):
        if t % rebal == 0:
            continue
        expected = _drift_hold(w.iloc[t - 1].to_numpy(), rets[t])
        assert np.allclose(w.iloc[t].to_numpy(), expected, atol=1e-14)
        checked += 1
    assert checked > 10
    assert w.sum(axis=1).max() <= 1.0 + EPS               # 漂移后仍不加杠杆


def test_mom12m_uses_skipped_window_not_recent_reversal():
    """12-1 语义：动量 = p[t-skip]/p[t-skip-lookback]-1，近端 skip 期的暴涨不会污染信号。"""
    n = 400
    steady = np.full(n, 0.001)                               # 稳步上行
    spike = np.full(n, -0.0005)                              # 长期阴跌
    spike[-10:] = 0.05                                       # 仅近端 10 期暴涨（应被 skip 剔除）
    idx = pd.bdate_range("2016-01-01", periods=n)
    px = pd.DataFrame({"STEADY": 100.0 * np.cumprod(1.0 + steady),
                       "SPIKE": 100.0 * np.cumprod(1.0 + spike)}, index=idx)
    data = MarketData(prices=px, periods_per_year=252)
    s = Mom12mTaaStrategy()
    mom = _trailing_mom(px, int(s.params["lookback"]), int(s.params["skip"]))
    naive = px.pct_change(int(s.params["lookback"]))          # 未跳过近端的「假 12 月动量」
    assert mom["STEADY"].iloc[-1] > 0.0 > mom["SPIKE"].iloc[-1]   # 跳过近端 → SPIKE 仍为负
    assert naive["SPIKE"].iloc[-1] > naive["STEADY"].iloc[-1]     # 不跳过则 SPIKE 反而最强
    w = s.generate_weights(data)
    assert w.iloc[-1]["STEADY"] > 0.0
    assert (w["SPIKE"].to_numpy() == 0.0).all()               # 全程不碰近端暴涨的资产
    assert _exposure(w).iloc[-1] <= 1.0 + EPS
