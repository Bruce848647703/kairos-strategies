"""riskmgmt 渠道测试：契约（形状/边界/确定性/无未来函数）+ 五类风控 overlay 的行为断言。

不调用 ``ks.discover()``：直接 import 本渠道模块与策略类。
"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.riskmgmt import (
    CHANNEL,
    STRATEGY_NAMES,
    CircuitBreakerStrategy,
    CppiStrategy,
    DrawdownThrottleStrategy,
    PortfolioVolTargetStrategy,
    TrailingStopOverlayStrategy,
    _benchmark_nav,
)

EPS = 1e-9
ALL = [CppiStrategy, DrawdownThrottleStrategy, TrailingStopOverlayStrategy,
       PortfolioVolTargetStrategy, CircuitBreakerStrategy]
NAMES = {"cppi", "drawdown_throttle", "trailing_stop_overlay",
         "portfolio_vol_target", "circuit_breaker"}
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
def _piecewise_market(segments, n_assets: int = 4, start: str = "2018-01-01") -> MarketData:
    """分段常数收益的确定性行情：segments = [(天数, 单期收益), ...]，所有资产同步。"""
    r = np.concatenate([np.full(int(n), float(v)) for n, v in segments])
    idx = pd.bdate_range(start, periods=len(r))
    px = 100.0 * np.cumprod(1.0 + r)
    cols = {"A%d" % i: px.copy() for i in range(n_assets)}
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252)


def _steady_market(n: int = 300, r: float = 0.001, n_assets: int = 4) -> MarketData:
    """平稳上行行情（常数正收益）：无回撤、无亏损、低波动。"""
    return _piecewise_market([(n, r)], n_assets=n_assets)


def _hard_crash_market() -> MarketData:
    """CPPI 场景：稳步上涨 → 连续 -30% 崩盘（击穿 floor）→ 暴力反弹（基准净值收复）。"""
    return _piecewise_market([(150, 0.002), (5, -0.30), (100, 0.08)])


def _gentle_crash_market() -> MarketData:
    """缓跌场景：上涨 → -2%/日 阴跌（回撤 >25%、跌破止损线）→ 横盘（永不收复）。"""
    return _piecewise_market([(150, 0.002), (60, -0.02), (80, 0.0)])


def _stop_recover_market() -> MarketData:
    """止损后修复场景：上涨 → 阴跌触发止损 → 强反弹并创出冻结高点 ×1.02 之上的新高。"""
    return _piecewise_market([(150, 0.002), (30, -0.02), (80, 0.02)])


def _breaker_market() -> MarketData:
    """熔断场景：上涨 → -3%/日 崩盘 25 天 → 长期横盘（亏损指标恢复、冷却结束）。"""
    return _piecewise_market([(120, 0.001), (25, -0.03), (100, 0.0)])


def _vol_regime_market(n_low: int = 260, n_high: int = 90, low_sigma: float = 0.0015,
                       high_sigma: float = 0.045, drift: float = 0.0004, seed: int = 7,
                       n_assets: int = 4):
    """前低波动、后高波动的合成行情（固定 seed，确定性）。"""
    rng = np.random.default_rng(seed)
    n = n_low + n_high
    sigma = np.concatenate([np.full(n_low, low_sigma), np.full(n_high, high_sigma)])
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = {}
    for i in range(n_assets):
        r = drift * (1.0 + 0.1 * i) + rng.standard_normal(n) * sigma
        cols["A%d" % i] = 100.0 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252), n_low


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
    assert (w.to_numpy() >= 0.0).all()                          # long_only
    assert np.abs(w.to_numpy()).sum(axis=1).max() <= 1.0 + EPS  # 每行和 ≤ 1（不加杠杆）
    assert np.abs(w.to_numpy()).max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", ALL)
def test_deterministic_across_calls_and_instances(cls):
    data = make_synthetic_universe(n_assets=5, n_days=320, seed=3)
    s = cls()
    a, b = s.generate_weights(data), s.generate_weights(data)
    pd.testing.assert_frame_equal(a, b)                         # 同实例两次一致
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
def test_weights_are_equal_weight_overlay(cls):
    """组合层 overlay：每行内各资产权重相同 = 总敞口 / N（基础敞口 = 等权 1/N × e）。"""
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=4)
    w = cls().generate_weights(data)
    n = data.n_assets
    expo = _exposure(w).to_numpy()
    assert np.allclose(w.to_numpy(), expo[:, None] / n, atol=1e-12)


@pytest.mark.parametrize("cls", ALL)
def test_meta_complete(cls):
    s = cls()                                                   # 必须无参可构造
    m = s.meta()
    assert m["channel"] == "riskmgmt" == CHANNEL
    assert m["universe"] == "overlay"
    assert m["long_only"] is True
    assert m["name"] in NAMES
    for k in ("description", "hypothesis", "source"):
        assert isinstance(m[k], str) and len(m[k]) > 10
    assert isinstance(m["params"], dict) and m["params"]


def test_names_unique_and_no_collision_with_existing_channels():
    got = [cls().name for cls in ALL]
    assert len(got) == len(set(got)) == 5
    assert set(got) == NAMES == set(STRATEGY_NAMES)
    assert not (NAMES & EXISTING_NAMES)                         # 不与既有策略重名


def test_steady_market_keeps_full_exposure():
    """平稳上行行情：五个 overlay 在各自预热期后都基本维持满仓（e = 1）。"""
    data = _steady_market()
    warmup = {                                  # 各策略「统计量未知不建仓」的预热长度
        "cppi": 0, "drawdown_throttle": 0, "trailing_stop_overlay": 0,
        "portfolio_vol_target": int(PortfolioVolTargetStrategy.params["span"]) - 1,
        "circuit_breaker": int(CircuitBreakerStrategy.params["window"]),
    }
    for cls in ALL:
        s = cls()
        expo = _exposure(s.generate_weights(data))
        w = warmup[s.name]
        assert expo.iloc[:w].abs().sum() == 0.0                 # 预热期全现金
        assert expo.iloc[w:].min() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- 行为断言
def test_cppi_full_exposure_then_derisk_and_lock_out_after_floor_breach():
    """CPPI：上涨期满仓；崩盘中先线性降敞口，净值击穿 floor 后**永久**空仓。

    手推递推：崩盘前 e=1、策略净值 V = 基准净值；首个崩盘日 V×0.7，
    e = min(1, 5×(V-0.8)/V) ∈ (0,1)；次日 V×(1-0.3e) ≤ 0.8 → 击穿 → e=0 且 V 冻结，
    此后即便基准净值反弹到远高于 floor 也不回来（吸收态、路径依赖）。
    """
    data = _hard_crash_market()
    nav = _benchmark_nav(data)
    expo = _exposure(CppiStrategy().generate_weights(data))
    crash0 = 150                                            # 首个崩盘日的行号
    v_pre = float(nav.iloc[crash0 - 1])                     # 崩盘前 V = 基准净值（e 恒为 1）
    assert expo.iloc[:crash0].min() == pytest.approx(1.0, abs=1e-12)   # 上涨期满仓
    v1 = v_pre * (1.0 + 1.0 * (-0.30))                      # 首个崩盘日净值
    e1 = min(1.0, 5.0 * (v1 - 0.8) / v1)
    assert 0.0 < e1 < 1.0                                   # 崩盘中先降杠杆（未满仓）
    assert expo.iloc[crash0] == pytest.approx(e1, abs=1e-12)
    v2 = v1 * (1.0 + e1 * (-0.30))                          # 次日击穿 floor
    assert v2 <= 0.8
    assert (expo.iloc[crash0 + 1:].to_numpy() == 0.0).all()  # 击穿后永久空仓
    assert float(nav.iloc[-1]) > 0.8                        # 基准净值已反弹回 floor 之上
    assert expo.iloc[-1] == 0.0                             # 但策略锁定在现金（吸收态）


def test_drawdown_throttle_cuts_exposure_as_drawdown_deepens():
    """回撤节流：回撤越深敞口越低（线性），回撤 ≥ max_dd 后为 0，未收复则一直空仓。"""
    data = _gentle_crash_market()
    p = DrawdownThrottleStrategy.params
    nav = _benchmark_nav(data)
    peak = nav.cummax()
    dd = 1.0 - nav / peak
    expo = _exposure(DrawdownThrottleStrategy().generate_weights(data))
    assert expo.iloc[:150].min() == pytest.approx(1.0, abs=1e-12)     # 上涨期无回撤 → 满仓
    crash = expo.iloc[150:165]                                        # 触及 max_dd 前的阴跌段
    assert (crash.diff().dropna() < 0.0).all()                        # 阴跌期敞口逐日下降
    deep = dd >= float(p["max_dd"])
    assert deep.any()
    assert (expo[deep].to_numpy() == 0.0).all()                       # 深回撤 → 空仓
    assert (expo.iloc[-80:].to_numpy() == 0.0).all()                  # 横盘未收复 → 仍空仓
    expected = np.clip(1.0 - dd / float(p["max_dd"]), 0.0, 1.0)       # 线性节流公式核对
    assert np.allclose(expo.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_trailing_stop_flattens_after_stop_and_stays_out():
    """移动止损：净值跌破 历史高点×(1-stop) 当日清仓，此后（未创新高）一直空仓。"""
    data = _gentle_crash_market()
    stop = float(TrailingStopOverlayStrategy.params["stop"])
    nav = _benchmark_nav(data).to_numpy()
    peak = np.maximum.accumulate(nav)
    t_stop = int(np.argmax(nav < peak * (1.0 - stop)))       # 首个触发日（独立手算）
    assert nav[t_stop - 1] >= peak[t_stop - 1] * (1.0 - stop)  # 前一日尚未触发
    expo = _exposure(TrailingStopOverlayStrategy().generate_weights(data))
    assert (expo.iloc[:t_stop].to_numpy() == 1.0).all()      # 触发前满仓
    assert (expo.iloc[t_stop:].to_numpy() == 0.0).all()      # 触发当日起清仓且不再回来


def test_trailing_stop_reenters_only_after_new_high_with_margin():
    """状态机再入场：止损后须创出 冻结高点×(1+reentry) 的新高才恢复满仓。"""
    data = _stop_recover_market()
    p = TrailingStopOverlayStrategy.params
    stop, reentry = float(p["stop"]), float(p["reentry"])
    nav = _benchmark_nav(data).to_numpy()
    peak = np.maximum.accumulate(nav)
    t_stop = int(np.argmax(nav < peak * (1.0 - stop)))
    frozen = peak[:t_stop + 1].max()                         # 冻结的历史高点
    after = np.nonzero(nav[t_stop:] > frozen * (1.0 + reentry))[0]
    assert len(after) > 0                                    # 反弹确实创出带缓冲的新高
    t_back = t_stop + int(after[0])
    expo = _exposure(TrailingStopOverlayStrategy().generate_weights(data))
    assert (expo.iloc[:t_stop].to_numpy() == 1.0).all()      # 止损前满仓
    assert (expo.iloc[t_stop:t_back].to_numpy() == 0.0).all()  # 场外全程现金
    assert (expo.iloc[t_back:].to_numpy() == 1.0).all()      # 新高后再入场并持有
    flips = int((np.diff(expo.to_numpy()) != 0).sum())
    assert flips == 2                                        # 恰好一次离场 + 一次再入场


def test_portfolio_vol_target_scales_with_realized_vol():
    """波动目标：低波段满仓（触顶不加杠杆），高波段大幅降仓；敞口 = 目标/已实现波动。"""
    data, n_low = _vol_regime_market()
    p = PortfolioVolTargetStrategy.params
    expo = _exposure(PortfolioVolTargetStrategy().generate_weights(data))
    low = expo.iloc[80:n_low]
    high = expo.iloc[n_low + 30:]
    assert low.mean() == pytest.approx(1.0, abs=1e-9)        # 低波：目标 > 已实现 → 满仓
    assert high.mean() < 0.5 < low.mean()                    # 高波：显著降仓
    assert expo.max() <= 1.0 + EPS
    # 精确性：e = clip(target / EWMA 已实现波动, 0, 1)，波动未知（预热）→ 空仓
    r = data.returns(1).mean(axis=1)
    var = r.pow(2).ewm(span=int(p["span"]), adjust=False,
                       min_periods=int(p["span"])).mean()
    rv = np.sqrt(var * 252.0).clip(lower=float(p["vol_floor"]))
    expected = (float(p["target_vol"]) / rv).clip(lower=float(p["min_scale"]),
                                                  upper=float(p["max_scale"]))
    ok = rv.notna()
    assert (expo[~ok].to_numpy() == 0.0).all()
    assert np.allclose(expo[ok].to_numpy(), expected[ok].to_numpy(), atol=1e-12)


def test_circuit_breaker_trips_cooldowns_and_recovers():
    """熔断：滚动亏损触发当日空仓 + 冷却 N 期；冷却结束且指标恢复后重新满仓。"""
    data = _breaker_market()
    p = CircuitBreakerStrategy.params
    window, thr, cool_n = int(p["window"]), float(p["loss_threshold"]), int(p["cooldown"])
    nav = _benchmark_nav(data).to_numpy()
    roll_loss = np.full(len(nav), np.nan)
    roll_loss[window:] = 1.0 - nav[window:] / nav[:-window]
    t_trip = int(np.argmax(np.nan_to_num(roll_loss, nan=-1.0) >= thr))  # 首个触发日（独立手算）
    expo = _exposure(CircuitBreakerStrategy().generate_weights(data))
    assert (expo.iloc[:window].to_numpy() == 0.0).all()      # 窗口预热期不建仓
    assert (expo.iloc[window:t_trip].to_numpy() == 1.0).all()  # 触发前满仓
    assert expo.iloc[t_trip] == 0.0                          # 熔断当日空仓
    assert (expo.iloc[t_trip:t_trip + cool_n + 1].to_numpy() == 0.0).all()  # 冷却期全程空仓
    assert cool_n + 1 <= int((expo.to_numpy() == 0.0).sum())  # 确有完整的冷却空仓段
    assert (expo.iloc[-50:].to_numpy() == 1.0).all()         # 横盘恢复 + 冷却结束 → 重新满仓
    assert expo.iloc[-1] == pytest.approx(1.0, abs=1e-12)
