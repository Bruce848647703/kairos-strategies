"""regime 渠道测试：契约（形状/边界/确定性/防未来）+ 四个状态切换策略的行为断言。

不调用 ``ks.discover()``：直接 import 本渠道模块与策略类。
"""
import time

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.regime import (
    STRATEGY_NAMES,
    BetaTimingStrategy,
    EmRegimeStrategy,
    RegimeVolTimingStrategy,
    TrendRegimeSwitchStrategy,
    _efficiency_ratio,
    _em_fit_gauss2,
    _em_posterior,
    _hysteresis_state,
    _rolling_percentile,
)

EPS = 1e-9
ALL = [RegimeVolTimingStrategy, TrendRegimeSwitchStrategy, EmRegimeStrategy,
       BetaTimingStrategy]
NAMES = {"regime_vol_timing", "trend_regime_switch", "em_regime", "beta_timing"}
# 仓库既有策略名（硬编码，避免调用 discover()）：本渠道不得与之重名
EXISTING_NAMES = frozenset({
    "adx_trend", "atr_breakout", "basket_neutral", "bollinger_breakout",
    "bollinger_reversion", "carry_proxy", "channel_atr_breakout", "coint_pairs",
    "crypto_momentum_247", "dca", "donchian_turtle", "dual_momentum",
    "dual_momentum_taa", "dual_thrust", "eof_stat_arb", "equal_weight_buy_hold",
    "equal_weight_ensemble", "equal_weight_rebal", "gbm_alpha", "grid_trading",
    "high_52w", "idio_momentum", "inverse_vol", "inverse_vol_ensemble",
    "keltner_breakout", "knn_alpha", "liquidity_premium", "logit_signal",
    "low_volatility", "lowvol_ls", "ma_deviation", "ma_ribbon", "macd_trend",
    "max_diversification", "min_variance_alloc", "mom_12m_taa", "month_of_year",
    "pairs_spread", "quality_ls", "range_breakout", "residual_momentum_ls",
    "reversal_ls", "ridge_alpha", "risk_parity_alloc", "rsi_reversion",
    "sharpe_weighted_ensemble", "short_term_reversal", "sma_cross",
    "trend_quality", "trend_regime_taa", "trend_reversion_blend", "ts_momentum",
    "tsmom_volscaled", "turn_of_month", "turtle_atr", "vol_regime_filter",
    "vol_scaled_momentum", "vol_target", "vol_target_taa", "volatility_breakout",
    "volume_imbalance", "volume_price_divergence", "vwap_reversion",
    "weekday_effect", "xs_momentum", "xs_momentum_ls", "xs_zscore_reversion",
    "zscore_reversion",
})


# ---------------------------------------------------------------- 数据构造
def _low_then_high_vol_market(n_low: int = 300, n_high: int = 200,
                              low_sigma: float = 0.0015, high_sigma: float = 0.05,
                              drift: float = 0.0002, seed: int = 11, n_assets: int = 4):
    """先平稳低波、后剧烈高波的确定性合成行情（固定 seed）。"""
    rng = np.random.default_rng(seed)
    n = n_low + n_high
    sigma = np.concatenate([np.full(n_low, low_sigma), np.full(n_high, high_sigma)])
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        r = drift * (1.0 + 0.1 * i) + rng.standard_normal(n) * sigma
        cols["A%d" % i] = 100.0 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_low


def _trend_range_trend_market(n_assets: int = 3, n1: int = 150, n2: int = 240,
                              n3: int = 120, slope: float = 0.004, amp: float = 0.05,
                              period: int = 8):
    """三段行情：稳定上行趋势 → 正弦震荡（周期整除 z 窗口）→ 稳定下行趋势。

    震荡段对数价格 = 水平位 + amp×sin(2πδ/period)，δ 为段内日序号：
    波谷在 δ ≡ period×3/4 (period=8 → δ≡6 mod 8)，波峰在 δ ≡ period/4 (δ≡2 mod 8)。
    """
    idx = pd.bdate_range("2018-01-01", periods=n1 + n2 + n3)
    d2 = np.arange(n2)
    cols = {}
    for i in range(n_assets):
        k = 1.0 + 0.15 * i
        a = amp - 0.005 * i
        seg1 = np.log(100.0) + np.cumsum(np.full(n1, slope * k))
        seg2 = seg1[-1] + a * np.sin(2.0 * np.pi * d2 / float(period))
        seg3 = seg2[-1] + np.cumsum(np.full(n3, -slope * k))
        cols["A%d" % i] = np.exp(np.concatenate([seg1, seg2, seg3]))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), (n1, n2)


def _bull_bear_em_market(n_bull: int = 500, n_bear: int = 300, n_assets: int = 3,
                         seed: int = 17):
    """两段不同均值/方差的高斯收益：前段 bull（正均值低波），后段 bear（负均值高波）。"""
    rng = np.random.default_rng(seed)
    base = np.concatenate([rng.normal(0.0015, 0.004, n_bull),
                           rng.normal(-0.008, 0.02, n_bear)])
    idx = pd.bdate_range("2016-01-01", periods=len(base))
    cols = {"A%d" % i: 100.0 * np.exp(np.cumsum(base * (1.0 + 0.05 * i)))
            for i in range(n_assets)}
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_bull


def _factor_beta_market(n: int = 300, seed: int = 7, betas=(0.5, 1.0, 1.5, 2.0)):
    """单因子市场：r_i = b_i × f（无异质噪声），beta 结构解析可验证。"""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {"A%d" % i: 100.0 * np.cumprod(1.0 + b * f) for i, b in enumerate(betas)}
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), list(betas)


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
    assert np.isfinite(w.to_numpy()).all()


@pytest.mark.parametrize("cls", ALL)
def test_long_only_and_row_sum_cap(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    w = cls().generate_weights(data)
    assert (w.to_numpy() >= 0.0).all()                            # long_only
    assert np.abs(w.to_numpy()).sum(axis=1).max() <= 1.0 + EPS    # 每行和 ≤ 1
    assert np.abs(w.to_numpy()).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic_across_calls_and_instances(cls):
    data = make_synthetic_universe(n_assets=5, n_days=500, seed=3)
    s = cls()
    a, b = s.generate_weights(data), s.generate_weights(data)
    pd.testing.assert_frame_equal(a, b)                           # 同实例两次一致
    pd.testing.assert_frame_equal(a, cls().generate_weights(data))  # 新实例一致


@pytest.mark.parametrize("cls", ALL)
def test_no_lookahead_future_prices_do_not_change_history(cls):
    """防未来：篡改 t 之后的价格，t 及之前的权重必须逐位不变（EM/beta 尤其严格）。"""
    data = make_synthetic_universe(n_assets=5, n_days=700, seed=5)
    t = 500
    base = cls().generate_weights(data)
    assert base.iloc[:t + 1].to_numpy().sum() > 0.0               # 前段确有非零权重（非空断言）
    p2 = data.prices.copy()
    ramp = np.linspace(1.0, 2.5, len(p2) - t - 1)[:, None]
    p2.iloc[t + 1:] = p2.iloc[t + 1:].to_numpy() * ramp           # 篡改 t 之后的价格
    mutated = cls().generate_weights(MarketData(prices=p2, periods_per_year=252))
    pd.testing.assert_frame_equal(base.iloc[:t + 1], mutated.iloc[:t + 1])


@pytest.mark.parametrize("cls", ALL)
def test_warmup_row0_zero(cls):
    """首期无任何历史统计量 → 不建仓（不用未来数据回填预热期）。"""
    data = make_synthetic_universe(n_assets=4, n_days=300, seed=9)
    w = cls().generate_weights(data)
    assert w.iloc[0].abs().sum() == 0.0


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                                                     # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "regime"
    assert m["long_only"] is True
    assert m["name"] in NAMES
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique_and_no_collision_with_existing_channels():
    got = [cls().name for cls in ALL]
    assert len(got) == len(set(got)) == 4
    assert set(got) == NAMES == set(STRATEGY_NAMES)
    assert not (NAMES & EXISTING_NAMES)                           # 不与既有策略重名


# ---------------------------------------------------------------- 辅助函数
def test_rolling_percentile_monotone_and_midrank():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.Series(np.arange(60, dtype=float), index=idx)
    down = up.iloc[::-1].reset_index(drop=True)
    down.index = idx
    p_up = _rolling_percentile(up, 20, 10)
    p_dn = _rolling_percentile(down, 20, 10)
    assert (p_up.dropna() == 1.0).all()                           # 严格递增 → 分位 1
    assert (p_dn.dropna() == 0.0).all()                           # 严格递减 → 分位 0
    const = pd.Series(3.0, index=idx)
    p_c = _rolling_percentile(const, 20, 10)
    assert (p_c.dropna() == 0.5).all()                            # 常数 → mid-rank 0.5
    assert p_up.iloc[:10].isna().all()                            # min_obs 预热 → NaN


def test_hysteresis_state_machine_holds_between_triggers():
    idx = pd.bdate_range("2020-01-01", periods=8)
    col = ["X"]
    enter = pd.DataFrame([[False], [True], [False], [False],
                          [False], [True], [False], [False]], index=idx, columns=col)
    exit_ = pd.DataFrame([[False], [False], [False], [True],
                          [False], [False], [False], [False]], index=idx, columns=col)
    st = _hysteresis_state(enter, exit_)["X"].to_numpy()
    assert list(st) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]   # 置位后保持，离场清零


def test_efficiency_ratio_separates_trend_and_range():
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n, dtype=float)
    trend = pd.DataFrame({"T": 100.0 * np.exp(0.004 * t)}, index=idx)
    osc = pd.DataFrame({"O": 100.0 * np.exp(0.05 * np.sin(2 * np.pi * t / 8.0))}, index=idx)
    er_t = _efficiency_ratio(trend, 20)["T"].iloc[40:]
    er_o = _efficiency_ratio(osc, 20)["O"].iloc[40:]
    assert (er_t > 0.9).all()                                     # 单边趋势 ER → 1
    assert (er_o < 0.35).all()                                    # 来回震荡 ER → 小


def test_em_fit_gauss2_recovers_components_deterministically():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(-0.01, 0.005, 400), rng.normal(0.01, 0.005, 400)])
    pi, mu, var = _em_fit_gauss2(x)
    pi2, mu2, var2 = _em_fit_gauss2(x)                            # 确定性：两次拟合一致
    assert np.array_equal(pi, pi2) and np.array_equal(mu, mu2) and np.array_equal(var, var2)
    assert np.isclose(pi.sum(), 1.0)
    assert np.isclose(np.sort(mu)[0], -0.01, atol=2e-3)           # 恢复两分量均值
    assert np.isclose(np.sort(mu)[1], 0.01, atol=2e-3)
    bull = int(np.argmax(mu))
    assert _em_posterior(np.array([0.02]), pi, mu, var, bull)[0] > 0.95   # 高收益 → bull
    assert _em_posterior(np.array([-0.02]), pi, mu, var, bull)[0] < 0.05  # 低收益 → bear
    deg = np.zeros(50)                                            # 退化输入不崩溃
    pi_d, mu_d, var_d = _em_fit_gauss2(deg)
    assert np.isfinite(pi_d).all() and np.isfinite(mu_d).all() and (var_d > 0).all()


# ---------------------------------------------------------------- 行为断言
def test_regime_vol_timing_deleverages_in_high_vol_regime():
    """先平稳低波后剧烈高波：高波段总敞口显著更低；缩放连续（多档，不是 0/1 开关）。"""
    data, n_low = _low_then_high_vol_market()
    w = RegimeVolTimingStrategy().generate_weights(data)
    expo = _exposure(w)
    low = expo.iloc[100:n_low - 5].mean()
    high = expo.iloc[n_low + 5:n_low + 75].mean()
    assert low > 0.6                                              # 低波段接近满仓
    assert high < 0.45                                            # 高波段深度降杠杆
    assert high < low - 0.25
    assert expo.max() <= 1.0 + EPS
    assert expo[expo > 0].nunique() > 5                           # 连续缩放而非二元开关


def test_trend_regime_switch_holds_momentum_in_uptrend():
    """趋势段：ER 判为趋势态，动量子逻辑持有强势资产（权重恒 = 1/N 满预算）。"""
    data, (n1, n2) = _trend_range_trend_market()
    w = TrendRegimeSwitchStrategy().generate_weights(data)
    per = 1.0 / data.n_assets
    seg1 = w.iloc[45:n1 - 1].to_numpy()
    assert np.allclose(seg1, per, atol=1e-12)                     # 上行趋势：全程持有
    close = data.prices.astype("float64")
    er = _efficiency_ratio(close, 20).rolling(3, min_periods=3).mean()
    trend = _hysteresis_state(er > 0.5, er < 0.25)
    assert (trend.iloc[45:n1 - 1].to_numpy() == 1.0).all()        # 状态机判为趋势态


def test_trend_regime_switch_buys_oversold_dips_in_range():
    """震荡段：ER 判为震荡态，均值回归子逻辑——波谷（超跌）持有、波峰（回归后）空仓。"""
    data, (n1, n2) = _trend_range_trend_market()
    w = TrendRegimeSwitchStrategy().generate_weights(data)
    per = 1.0 / data.n_assets
    vals = w.to_numpy()
    troughs = [n1 + 6 + 8 * k for k in range(3, 28) if n1 + 6 + 8 * k < n1 + n2 - 2]
    peaks = [n1 + 2 + 8 * k for k in range(3, 28) if n1 + 2 + 8 * k < n1 + n2 - 2]
    assert troughs and peaks
    assert np.allclose(vals[troughs], per, atol=1e-12)            # 波谷超跌 → 买入持有
    assert np.allclose(vals[peaks], 0.0, atol=1e-12)              # 回归均值上方 → 离场
    close = data.prices.astype("float64")
    er = _efficiency_ratio(close, 20).rolling(3, min_periods=3).mean()
    trend = _hysteresis_state(er > 0.5, er < 0.25)
    assert (trend.iloc[n1 + 25:n1 + n2].to_numpy() == 0.0).all()  # 状态机判为震荡态


def test_trend_regime_switch_refuses_negative_momentum_in_downtrend():
    """下行趋势段：趋势态 + 负动量 → 不持有（动量子逻辑拒绝接刀），权重全 0。"""
    data, (n1, n2) = _trend_range_trend_market()
    w = TrendRegimeSwitchStrategy().generate_weights(data)
    s3 = n1 + n2
    assert (w.iloc[s3 + 40:].to_numpy() == 0.0).all()
    close = data.prices.astype("float64")
    er = _efficiency_ratio(close, 20).rolling(3, min_periods=3).mean()
    trend = _hysteresis_state(er > 0.5, er < 0.25)
    assert (trend.iloc[s3 + 40:].to_numpy() == 1.0).all()         # 确为趋势态（非震荡态买跌）


def test_em_regime_detects_bear_and_deleverages():
    """两段不同均值/方差收益：EM 识别 bear 状态并降杠杆（bull 段敞口显著更高）。"""
    data, n_bull = _bull_bear_em_market()
    w = EmRegimeStrategy().generate_weights(data)
    expo = _exposure(w)
    min_train = EmRegimeStrategy.params["min_train"]
    assert (expo.iloc[:min_train].to_numpy() == 0.0).all()        # 预热期全现金
    bull = expo.iloc[n_bull - 200:n_bull].mean()
    bear = expo.iloc[-100:].mean()
    assert bull > 0.35                                            # bull 段维持敞口
    assert bear < 0.3                                             # bear 段显著降杠杆
    assert bull - bear > 0.15
    mkt = data.returns(1).mean(axis=1)
    tail_ret = mkt.iloc[-100:]
    worst = expo.iloc[-100:][tail_ret <= tail_ret.quantile(0.2)]
    assert worst.mean() < 0.05                                    # 最深 bear 收益 → 近空仓
    assert expo.max() <= 1.0 + EPS and (expo.to_numpy() >= 0.0).all()


def test_beta_timing_allocates_inverse_to_beta():
    """单因子市场（b=[0.5,1,1.5,2]）：权重 = clip(2−0.8b)/N = [0.4,0.3,0.2,0.1]，高 beta 降敞口。"""
    data, betas = _factor_beta_market()
    w = BetaTimingStrategy().generate_weights(data)
    W = BetaTimingStrategy.params["window"]
    expected = np.array([(2.0 - 0.8 * b) / 4.0 for b in betas])   # beta_measured = b/mean(b)
    rows = w.iloc[W + 5:W + 50].to_numpy()
    assert np.allclose(rows, expected, atol=1e-9)                 # 解析可验证的精确权重
    last = w.iloc[-1]
    assert last["A0"] > last["A1"] > last["A2"] > last["A3"]      # beta 越高权重越低
    sums = w.iloc[W:].sum(axis=1)
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)           # 组合层行和 ≈ 1（≤1）
    assert (w.iloc[:W - 1].to_numpy() == 0.0).all()               # beta 预热期不建仓


def test_em_regime_performance_within_budget():
    """性能：n_assets=8、n_days=1000 的 walk-forward EM 在数秒内完成。"""
    data = make_synthetic_universe(n_assets=8, n_days=1000, seed=1)
    t0 = time.time()
    w = EmRegimeStrategy().generate_weights(data)
    elapsed = time.time() - t0
    assert w.shape == (1000, 8)
    assert elapsed < 15.0
