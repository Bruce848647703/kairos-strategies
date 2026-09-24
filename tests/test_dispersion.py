"""dispersion 渠道测试：契约（形状/边界/确定性/防未来）+ 三个分散度策略的行为断言。

不调用 ``ks.discover()``：直接 import 本渠道模块与策略类。
"""
import time

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.dispersion import (
    STRATEGY_NAMES,
    CorrelationRegimeStrategy,
    DispersionTimingStrategy,
    VolDispersionLsStrategy,
    _avg_pairwise_corr,
    _dispersion_state,
    _inverse_vol_weights,
    _vol_dispersion_size,
)

EPS = 1e-9
ALL = [DispersionTimingStrategy, CorrelationRegimeStrategy, VolDispersionLsStrategy]
LONG_ONLY = [DispersionTimingStrategy, CorrelationRegimeStrategy]
LS = [VolDispersionLsStrategy]
NAMES = {"dispersion_timing", "correlation_regime", "vol_dispersion_ls"}
# 仓库既有策略名（硬编码，避免调用 discover()）：本渠道不得与之重名
EXISTING_NAMES = frozenset({
    "adx_trend", "atr_breakout", "avellaneda_stoikov_proxy", "basket_neutral",
    "beta_timing", "betting_against_beta", "bollinger_breakout",
    "bollinger_reversion", "carry_proxy", "channel_atr_breakout",
    "circuit_breaker", "clustered_inverse_vol", "coint_pairs",
    "coint_pairs_portfolio", "cppi", "crypto_momentum_247", "dca",
    "defensive_quality", "donchian_turtle", "drawdown_averse",
    "drawdown_throttle", "dual_momentum", "dual_momentum_taa", "dual_thrust",
    "em_regime", "eof_stat_arb", "equal_weight_buy_hold",
    "equal_weight_ensemble", "equal_weight_rebal", "equalweight_composite",
    "frog_in_pan", "gbm_alpha", "grid_mm_daily", "grid_trading",
    "group_momentum", "herc", "high_52w", "hrp", "ic_weighted_composite",
    "idio_momentum", "inventory_skew_mm", "inverse_vol",
    "inverse_vol_ensemble", "keltner_breakout", "knn_alpha", "liquidity_premium",
    "liquidity_provision_ls",     "logit_signal", "low_beta_timing",
    "low_volatility", "lowvol_ls", "ma_deviation", "ma_ribbon", "macd_trend",
    "max_diversification", "max_ir_composite",
    "min_variance_alloc", "mom_12m_taa", "momentum_spread_timing",
    "month_of_year", "pairs_spread", "pca_factor", "quality_ls",
    "range_breakout", "regime_vol_timing", "residual_momentum_ls",
    "reversal_ls", "ridge_alpha", "risk_parity_alloc", "rsi_reversion",
    "sector_neutral_pairs", "sharpe_weighted_ensemble", "short_term_reversal",
    "sma_cross", "ssd_pairs", "trailing_stop_overlay", "trend_quality",
    "trend_regime_switch", "trend_regime_taa", "trend_reversion_blend",
    "ts_momentum", "tsmom_volscaled", "turn_of_month", "turtle_atr",
    "vol_managed_momentum", "vol_regime_filter", "vol_scaled_momentum",
    "vol_target", "vol_target_taa", "volatility_breakout", "volume_imbalance",
    "volume_price_divergence", "vwap_reversion", "weekday_effect",
    "xs_momentum", "xs_momentum_ls", "xs_zscore_reversion",
    "zscore_reversion", "portfolio_vol_target",
})


# ---------------------------------------------------------------- 数据构造
def _prices_from_returns(r: np.ndarray, start: str = "2015-01-05",
                         p0: float = 100.0) -> pd.DataFrame:
    """由日收益矩阵精确构造价格面板：第 k+1 行价格的 pct_change 恰为 r[k]。"""
    r = np.asarray(r, dtype="float64")
    T, N = r.shape
    idx = pd.bdate_range(start, periods=T + 1)
    pad = np.vstack([np.zeros((1, N)), r])           # 首日为基期，收益从次日起
    px = p0 * np.cumprod(1.0 + pad, axis=0)
    return pd.DataFrame(px, index=idx, columns=["A%d" % i for i in range(N)])


def _dispersion_two_regime_market(n1: int = 250, n2: int = 200):
    """两段确定性行情：高分散（8 资产常数但分化的日漂移，4 涨 4 跌）→ 低分散（齐涨）。

    高分散段：截面收益标准差大（走势分化）、个体时序波动为 0（常数漂移）→ DR 极高；
    低分散段：所有资产收益完全相同 → 截面标准差 = 0 → DR = 0。
    """
    N = 8
    c = np.array([0.010, 0.005, 0.002, 0.0005,
                  -0.0005, -0.002, -0.005, -0.010])
    r1 = np.tile(c, (n1, 1))                         # 高分散：强者恒强、弱者恒弱
    r2 = np.tile(np.full(N, 0.002), (n2, 1))         # 低分散：齐涨（完全同步）
    prices = _prices_from_returns(np.vstack([r1, r2]))
    return MarketData(prices=prices, periods_per_year=252), n1


def _low_then_high_corr_market(n1: int = 260, n2: int = 260, seed: int = 23,
                               n_assets: int = 4):
    """两段行情：低相关（各资产独立噪声）→ 高相关（单一共同因子驱动，两两相关 ≈ 1）。"""
    rng = np.random.default_rng(seed)
    r1 = 0.0002 + 0.01 * rng.standard_normal((n1, n_assets))    # 独立 → ρ̂ ≈ 0
    f = 0.0003 + 0.012 * rng.standard_normal(n2)                # 共同冲击
    b = 1.0 + 0.15 * np.arange(n_assets)
    r2 = f[:, None] * b[None, :]                                # 同涨同跌 → ρ̂ ≈ 1
    prices = _prices_from_returns(np.vstack([r1, r2]))
    return MarketData(prices=prices, periods_per_year=252), n1


def _independent_heterovol_market(n: int = 360, seed: int = 31):
    """高分散行情：6 资产相互独立、日波动差异悬殊 → 单股波动远高于等权指数波动。"""
    rng = np.random.default_rng(seed)
    sig = np.array([0.004, 0.008, 0.012, 0.020, 0.030, 0.045])
    r = 0.0002 + rng.standard_normal((n, 6)) * sig[None, :]
    return MarketData(prices=_prices_from_returns(r), periods_per_year=252)


def _synced_market(n: int = 360, seed: int = 33, n_assets: int = 6):
    """同步行情：所有资产收益完全相同 → 单股波动 = 指数波动（分散度消失）。"""
    rng = np.random.default_rng(seed)
    f = 0.0002 + 0.012 * rng.standard_normal(n)
    r = np.tile(f, (n_assets, 1)).T
    return MarketData(prices=_prices_from_returns(r), periods_per_year=252)


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


@pytest.mark.parametrize("cls", LONG_ONLY)
def test_long_only_bounds(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    a = cls().generate_weights(data).to_numpy()
    assert (a >= 0.0).all()                               # long_only
    assert np.abs(a).sum(axis=1).max() <= 1.0 + EPS       # 每行和 ≤ 1
    assert np.abs(a).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", LS)
def test_ls_dollar_neutral_bounds(cls):
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=1)
    a = cls().generate_weights(data).to_numpy()
    assert np.abs(a.sum(axis=1)).max() <= 1e-9            # 美元中性：行和 ≈ 0
    assert np.abs(a).sum(axis=1).max() <= 1.0 + EPS       # 绝对值和 ≤ 1（无杠杆）
    assert np.abs(a).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic_across_calls_and_instances(cls):
    data = make_synthetic_universe(n_assets=5, n_days=500, seed=3)
    s = cls()
    a, b = s.generate_weights(data), s.generate_weights(data)
    pd.testing.assert_frame_equal(a, b)                   # 同实例两次一致
    pd.testing.assert_frame_equal(a, cls().generate_weights(data))  # 新实例一致


@pytest.mark.parametrize("cls", ALL)
def test_no_lookahead_future_prices_do_not_change_history(cls):
    """防未来：篡改 t 之后的价格，t 及之前的权重必须逐位不变。"""
    data = make_synthetic_universe(n_assets=5, n_days=700, seed=5)
    t = 500
    base = cls().generate_weights(data)
    assert np.abs(base.iloc[:t + 1].to_numpy()).sum() > 0.0   # 前段确有非零权重
    p2 = data.prices.copy()
    ramp = np.linspace(1.0, 2.5, len(p2) - t - 1)[:, None]
    p2.iloc[t + 1:] = p2.iloc[t + 1:].to_numpy() * ramp       # 篡改 t 之后的价格
    mutated = cls().generate_weights(MarketData(prices=p2, periods_per_year=252))
    pd.testing.assert_frame_equal(base.iloc[:t + 1], mutated.iloc[:t + 1])


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                                             # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "dispersion"
    assert m["name"] in NAMES
    assert m["long_only"] is (cls is not VolDispersionLsStrategy)
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique_and_no_collision_with_existing_channels():
    got = [cls().name for cls in ALL]
    assert len(got) == len(set(got)) == 3
    assert set(got) == NAMES == set(STRATEGY_NAMES)
    assert not (NAMES & EXISTING_NAMES)                   # 不与既有策略重名


def test_warmup_fallbacks():
    """预热期回退：分散度/相关性未知 → 等权；波动率分散度未确认 → 空仓。"""
    data = make_synthetic_universe(n_assets=4, n_days=300, seed=9)
    w_dt = DispersionTimingStrategy().generate_weights(data)
    assert np.allclose(w_dt.iloc[0].to_numpy(), 1.0 / 4.0)   # 等权回退
    w_cr = CorrelationRegimeStrategy().generate_weights(data)
    assert np.allclose(w_cr.iloc[0].to_numpy(), 1.0 / 4.0)   # 等权满仓回退
    w_ls = VolDispersionLsStrategy().generate_weights(data)
    assert (w_ls.iloc[:60].to_numpy() == 0.0).all()          # 空仓回退


# ---------------------------------------------------------------- 辅助函数
def test_avg_pairwise_corr_recovers_known_structure():
    n = 240
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=n)
    f = rng.standard_normal(n) * 0.01
    r = pd.DataFrame({"A0": f, "A1": 2.0 * f,
                      "A2": rng.standard_normal(n) * 0.01}, index=idx)
    c = _avg_pairwise_corr(r, 60)
    assert c.iloc[:59].isna().all()                       # 窗口不足 → NaN
    assert c.iloc[59:].notna().all()
    # 三对：(A0,A1)=1，(A0,A2)≈0，(A1,A2)≈0 → 平均 ≈ 1/3
    assert abs(c.iloc[100:].mean() - 1.0 / 3.0) < 0.15
    synced = pd.DataFrame(np.tile(f, (3, 1)).T, columns=["A0", "A1", "A2"],
                          index=idx)
    c2 = _avg_pairwise_corr(synced, 60)
    assert (c2.iloc[100:] > 0.999).all()                  # 完全同步 → ρ̂ ≈ 1


def test_dispersion_state_separates_synced_and_independent():
    n = 240
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["A%d" % i for i in range(4)]
    f = 0.0005 + 0.01 * rng.standard_normal(n)
    synced = pd.DataFrame(np.tile(f, (4, 1)).T, columns=cols, index=idx)
    indep = pd.DataFrame(0.01 * rng.standard_normal((n, 4)), columns=cols,
                         index=idx)
    d_s = _dispersion_state(synced, 252, 10, 40, 0.01, 0.30, 1.00)
    d_i = _dispersion_state(indep, 252, 10, 40, 0.01, 0.30, 1.00)
    assert np.nanmax(d_s.to_numpy()) < 1e-9               # 完全同步 → d = 0
    assert d_i.iloc[60:].mean() > 0.5                     # 独立波动 → 高分散状态
    assert np.nanmax(d_i.to_numpy()) <= 1.0               # 状态有界


def test_inverse_vol_weights_simplex_and_ordered():
    n = 300
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2020-01-01", periods=n)
    sig = np.array([0.005, 0.010, 0.020, 0.040])
    r = pd.DataFrame(rng.standard_normal((n, 4)) * sig[None, :],
                     columns=["A%d" % i for i in range(4)], index=idx)
    ivw = _inverse_vol_weights(r, 60, 252, 1e-4)
    rows = ivw.iloc[100:]
    assert np.allclose(rows.sum(axis=1).to_numpy(), 1.0, atol=1e-12)  # 单纯形
    assert (rows.to_numpy() > 0.0).all()
    m = rows.mean()
    assert m["A0"] > m["A1"] > m["A2"] > m["A3"]          # 波动越低权重越高


def test_vol_dispersion_size_gates_on_dispersion():
    p = VolDispersionLsStrategy.params
    hi = _independent_heterovol_market()
    lo = _synced_market()
    s_hi = _vol_dispersion_size(hi.prices.pct_change(), 252, p["vol_window"],
                                p["size_lo"], p["size_hi"], p["vol_floor"])
    s_lo = _vol_dispersion_size(lo.prices.pct_change(), 252, p["vol_window"],
                                p["size_lo"], p["size_hi"], p["vol_floor"])
    assert s_hi.iloc[-100:].min() > 0.99                  # 单股波动 >> 指数波动 → 满强度
    assert np.abs(s_lo.to_numpy()).max() == 0.0           # 完全同步 → 平仓（含预热 NaN→0）


# ---------------------------------------------------------------- 行为断言
def test_dispersion_timing_tilts_reversion_when_dispersed_index_when_synced():
    """高分散段倾斜选股/均值回归腿（做多近期弱者）；低分散段倾斜指数/趋势腿（等权持有）。"""
    data, n1 = _dispersion_two_regime_market()
    s = DispersionTimingStrategy()
    p = s.params
    w = s.generate_weights(data)
    n = data.n_assets
    hi = w.iloc[90:n1]                                    # 高分散段深处（A0..A3 涨、A4..A7 跌）
    lo = w.iloc[n1 + 80:]                                 # 低分散段深处（齐涨）
    hi_np = hi.to_numpy()
    assert (hi_np[:, :4] < 1e-12).all()                   # 高分散：强势(上涨)资产 ≈ 0
    assert (hi_np[:, 4:] > 0.0).all()                     # 弱势(下跌)资产全部获得正权重
    means = hi.mean()
    assert means["A7"] > means["A6"] > means["A5"] > means["A4"]   # 跌得越深回归权重越大
    assert np.allclose(lo.to_numpy(), 1.0 / n, atol=1e-12)         # 低分散：等权指数/趋势腿
    # 两段倾斜显著不同：高分散段权重与近期收益负相关（回归倾斜），低分散段完全均匀
    rr = data.prices.pct_change(int(p["rev_window"]))
    corr_hi = w.iloc[90:n1].corrwith(rr.iloc[90:n1], axis=1)
    assert corr_hi.mean() < -0.5
    assert w.iloc[n1 + 80:].std(axis=1).max() < 1e-12
    # 分散度状态本身：高分散段 ≈ 1、低分散段 ≈ 0
    d = _dispersion_state(data.prices.pct_change(), data.periods_per_year,
                          p["disp_window"], p["vol_window"], p["vol_floor"],
                          p["ratio_lo"], p["ratio_hi"])
    assert d.iloc[90:n1].mean() > 0.9
    assert d.iloc[n1 + 80:].mean() < 0.1


def test_correlation_regime_deleverages_when_correlation_spikes():
    """低相关段满仓等权分散；高相关段（危机同步）总敞口降到 expo_min 附近。"""
    data, n1 = _low_then_high_corr_market()
    s = CorrelationRegimeStrategy()
    w = s.generate_weights(data)
    expo = w.sum(axis=1)
    low = expo.iloc[80:n1].mean()
    high = expo.iloc[n1 + 80:].mean()
    assert low > 0.9                                      # 低相关 → 满仓分散
    assert high < 0.25                                    # 高相关 → risk-off 去杠杆
    assert high < low - 0.5
    assert abs(high - s.params["expo_min"]) < 0.03        # 逼近敞口下限
    arr = w.to_numpy()
    assert (arr >= 0.0).all()
    assert expo.max() <= 1.0 + EPS
    # 形态 = 等权 × 敞口缩放：行内严格等权
    assert np.allclose(arr, arr.mean(axis=1, keepdims=True), atol=1e-12)


def test_vol_dispersion_ls_longs_dispersed_leg_shorts_index_leg():
    """单股波动远高于指数波动（高分散）：满强度做空指数腿/做多分散腿，美元中性。"""
    data = _independent_heterovol_market()
    w = VolDispersionLsStrategy().generate_weights(data)
    tail = w.iloc[-120:]
    a = tail.to_numpy()
    assert np.abs(a.sum(axis=1)).max() <= 1e-9            # 行和 ≈ 0（美元中性）
    assert np.abs(a).sum(axis=1).max() <= 1.0 + EPS       # 绝对值和 ≤ 1
    col = tail.mean()
    assert col["A0"] > 0.08                               # 最低波动 → 分散腿净多头主力
    assert col["A5"] < -0.04                              # 最高波动 → 指数腿空头占优
    assert col["A0"] > col["A1"] > col["A3"] > col["A5"]  # 净权重随单股波动递减
    gross_long = tail.clip(lower=0.0).sum(axis=1)
    gross_short = (-tail).clip(lower=0.0).sum(axis=1)
    assert gross_long.mean() > 0.1                        # 两腿均实质在场
    assert np.abs(gross_long - gross_short).max() <= 1e-9  # 多空美元平衡


def test_vol_dispersion_ls_flattens_when_market_synced():
    """同步行情（单股波动 = 指数波动，分散度消失）→ 门控关闭，整体平仓。"""
    data = _synced_market()
    w = VolDispersionLsStrategy().generate_weights(data)
    assert np.abs(w.iloc[80:].to_numpy()).max() < 1e-12


# ---------------------------------------------------------------- 性能
def test_performance_full_universe():
    data = make_synthetic_universe(n_assets=8, n_days=1000, seed=1)
    t0 = time.time()
    for cls in ALL:
        w = cls().generate_weights(data)
        assert w.shape == (1000, 8)
    assert time.time() - t0 < 20.0
