"""ensemble 渠道测试。

覆盖：
  - 契约与数值健全：形状/索引列对齐/有限值、long_only 权重 >= 0、每行权重和 <= 1+eps、确定性；
  - 元信息：channel='ensemble'、名字唯一且不与基础策略重名、无参可构造、只 import 其它渠道公开类；
  - 行为断言（数值核对）：
      equal_weight_ensemble 权重 == 各基础策略权重面板的算术平均；
      trend_reversion_blend 权重 == 0.6·donchian_turtle + 0.4·zscore_reversion(多头腿)；
      inverse_vol / sharpe 的混合系数 == 测试内独立重算的 walk-forward 公式（shift 一期）；
      逆波动系数把最高权重给「截至 t-1 波动最低」的基础策略；负夏普基础策略系数恒为 0；
      预热期系数回退等权，且此时元策略权重与 equal_weight_ensemble 完全一致；
  - 防未来函数：篡改 t0 之后的价格，t0 及之前的元策略权重逐元素完全不变；
      截断样本重算，前缀权重/系数完全一致；基础策略回测收益 == 手工「上一期权重 × 本期收益」；
  - 性能预算与极小数据边界。
"""
import time

import numpy as np
import pandas as pd
import pytest

import kairos_strategies as ks
from kairos_strategies import MarketData, align_weights
from kairos_strategies.channels import ensemble as ens
from kairos_strategies.channels.ensemble import (
    EqualWeightEnsembleStrategy,
    InverseVolEnsembleStrategy,
    SharpeWeightedEnsembleStrategy,
    TrendReversionBlendStrategy,
)
from kairos_strategies.channels.meanrev import ZscoreReversionStrategy
from kairos_strategies.channels.technical import DonchianTurtleStrategy

STRATEGIES = [EqualWeightEnsembleStrategy, InverseVolEnsembleStrategy,
              SharpeWeightedEnsembleStrategy, TrendReversionBlendStrategy]
WEIGHTED = [InverseVolEnsembleStrategy, SharpeWeightedEnsembleStrategy]
EPS = 1e-9
N_ASSETS, N_DAYS, SEED = 6, 400, 1
T0 = 300                      # 防未来篡改点（之后价格被剧烈改写）
PREFIX = 250                  # 前缀截断长度


@pytest.fixture(scope="module")
def synth():
    return ks.make_synthetic_universe(n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)


def _tamper_future(data: MarketData, t0: int = T0, seed: int = 99) -> MarketData:
    """把 t0 之后的价格乘上 [2.5, 3.0) 的随机因子（剧烈改写未来，价格仍恒正）。"""
    rng = np.random.default_rng(seed)
    p = data.prices.copy()
    fut = p.iloc[t0 + 1:]
    p.iloc[t0 + 1:] = fut * (2.5 + 0.5 * rng.random(fut.shape))
    return MarketData(prices=p,
                      volumes=None if data.volumes is None else data.volumes.copy(),
                      periods_per_year=data.periods_per_year, name=data.name)


def _truncate(data: MarketData, m: int) -> MarketData:
    return MarketData(prices=data.prices.iloc[:m],
                      volumes=None if data.volumes is None else data.volumes.iloc[:m],
                      periods_per_year=data.periods_per_year, name=data.name)


def _core_panels(data: MarketData):
    """基础策略（核心池）的 long_only 权重面板，顺序与 ens.core_base_names() 一致。"""
    names = ens.core_base_names()
    panels = ens.base_weight_panels(data, names=names)
    return names, [panels[nm] for nm in names]


def _expected_coefficients(rets: pd.DataFrame, kind: str, window: int, min_obs: int,
                           floor: float, ppy: int) -> np.ndarray:
    """测试内独立重算的 walk-forward 混合系数（不复用被测模块的实现）。"""
    roll = rets.rolling(int(window), min_periods=int(min_obs))
    sd = roll.std(ddof=1)
    if kind == "sharpe":
        stat = (roll.mean() / sd.where(sd > floor)) * np.sqrt(float(ppy))
        raw = stat.clip(lower=0.0)
    else:
        raw = 1.0 / sd.clip(lower=float(floor))
    raw = raw.shift(1)                                    # 防未来：第 t 期只用 [0, t-1]
    vals = np.nan_to_num(raw.to_numpy(dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.clip(vals, 0.0, None)
    tot = vals.sum(axis=1)
    bad = tot <= 1e-18
    out = vals / np.where(bad, 1.0, tot)[:, None]
    out[bad] = 1.0 / vals.shape[1]                        # 预热/退化 -> 等权
    return out


# ---------------------------------------------------------------- 契约与数值健全

def test_shape_columns_index_alignment(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame), cls.name
        assert w.shape == (len(synth.dates), len(synth.symbols)), cls.name
        assert list(w.columns) == synth.symbols, cls.name
        assert w.index.equals(synth.dates), cls.name


def test_all_values_finite(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert np.isfinite(w.to_numpy()).all(), cls.name


def test_long_only_weights_non_negative(synth):
    for cls in STRATEGIES:
        s = cls()
        assert s.long_only is True
        w = s.generate_weights(synth)
        assert w.to_numpy().min() >= 0.0, cls.name


def test_row_sum_le_one(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
        assert w.to_numpy().max() <= 1.0 + EPS, cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        w1 = cls().generate_weights(synth)
        w2 = cls().generate_weights(synth)          # 新实例、二次调用
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.to_numpy(), w2.to_numpy()), cls.name


def test_no_arg_construction_and_meta_complete():
    names = set()
    base_names = {"sma_cross", "donchian_turtle", "ts_momentum", "low_volatility",
                  "inverse_vol", "zscore_reversion"}
    for cls in STRATEGIES:
        s = cls()                                   # 必须无参可构造
        meta = s.meta()
        assert s.channel == "ensemble"
        assert s.name and s.name == s.name.strip().lower()
        assert s.name not in names and s.name not in base_names, s.name
        names.add(s.name)
        assert s.long_only is True
        assert s.universe in ("timing", "cross_section")
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"], s.name


def test_meta_documents_base_strategies_and_weighting():
    """元信息必须说明组合了哪些基础策略与加权方式。"""
    pooled = [EqualWeightEnsembleStrategy, InverseVolEnsembleStrategy,
              SharpeWeightedEnsembleStrategy]
    for cls in pooled:
        s = cls()
        assert list(s.params["bases"]) == ens.core_base_names()
        text = s.description + s.hypothesis + s.source
        for nm in ens.core_base_names():
            assert nm in text or nm.replace("_", "") in text, (s.name, nm)
    b = TrendReversionBlendStrategy()
    assert b.params["trend_base"] == "donchian_turtle"
    assert b.params["reversion_base"] == "zscore_reversion"
    trend, rev, wt, wr = b.blend_coefficients()
    assert (trend, rev) == ("donchian_turtle", "zscore_reversion")
    assert wt >= 0.0 and wr >= 0.0
    assert abs(wt + wr - 1.0) <= EPS                # 混合系数非负且和≈1
    assert abs(wt - 0.6) <= EPS and abs(wr - 0.4) <= EPS


def test_only_public_base_classes_imported():
    """只允许 import 其它渠道的公开 Strategy 类：不得引入其私有下划线函数。"""
    prefix = "kairos_strategies.channels."
    foreign = {k: v for k, v in vars(ens).items()
               if getattr(v, "__module__", None) and
               str(getattr(v, "__module__")).startswith(prefix) and
               getattr(v, "__module__") != ens.__name__}
    assert foreign, "本渠道应复用其它渠道的基础策略"
    for nm, obj in foreign.items():
        assert not nm.startswith("_"), f"不应以私有名暴露外部符号: {nm}"
        assert isinstance(obj, type) and issubclass(obj, ks.Strategy), nm
        assert getattr(obj, "channel", "") != "ensemble", nm


def test_base_pool_is_expected_strategies():
    assert ens.core_base_names() == ["sma_cross", "donchian_turtle", "ts_momentum",
                                     "low_volatility", "inverse_vol"]
    strats = ens.base_strategies()
    assert list(strats) == ens.core_base_names() + ["zscore_reversion"]
    for nm, s in strats.items():
        assert s.name == nm
        assert s.channel in ("technical", "momentum", "factor", "allocation", "meanrev")


# ------------------------------------------------------- 基础面板 / 基础回测收益

def test_base_panels_are_long_only_and_aligned(synth):
    panels = ens.base_weight_panels(synth)
    assert set(panels) == set(ens.core_base_names()) | {"zscore_reversion"}
    for nm, p in panels.items():
        assert p.shape == (N_DAYS, N_ASSETS), nm
        assert list(p.columns) == synth.symbols, nm
        assert np.isfinite(p.to_numpy()).all(), nm
        assert p.to_numpy().min() >= 0.0, nm                    # 统一成 long_only
        assert p.sum(axis=1).max() <= 1.0 + EPS, nm


def test_base_returns_match_manual_lagged_computation(synth):
    """基础策略回测收益 == 手工「上一期权重 × 本期资产收益」（t 期期末已知）。"""
    rets = ens.base_backtest_returns(synth)
    names = ens.core_base_names()
    assert list(rets.columns) == names
    assert rets.index.equals(synth.dates)
    assert np.isfinite(rets.to_numpy()).all()
    asset_ret = synth.prices.pct_change().fillna(0.0)
    panels = ens.base_weight_panels(synth, names=names)
    for nm in names:
        manual = (panels[nm].shift(1).fillna(0.0) * asset_ret).sum(axis=1)
        assert np.allclose(rets[nm].to_numpy(), manual.to_numpy(), atol=1e-15), nm


# ------------------------------------------------------------------ 行为断言

def test_equal_weight_equals_mean_of_base_panels(synth):
    """equal_weight_ensemble 的权重 == 各基础策略权重面板的均值（数值核对）。"""
    names, panels = _core_panels(synth)
    expected = np.stack([p.to_numpy() for p in panels]).mean(axis=0)
    w = EqualWeightEnsembleStrategy().generate_weights(synth).to_numpy()
    assert np.allclose(w, expected, atol=1e-12, rtol=0.0)
    # 确实是「混合」而非退化成某一条腿
    for p in panels:
        assert not np.allclose(w, p.to_numpy(), atol=1e-6), names


def test_equal_weight_row_sums_are_mean_of_base_row_sums(synth):
    _, panels = _core_panels(synth)
    expected_rows = np.mean([p.sum(axis=1).to_numpy() for p in panels], axis=0)
    w = EqualWeightEnsembleStrategy().generate_weights(synth)
    assert np.allclose(w.sum(axis=1).to_numpy(), expected_rows, atol=1e-12)
    assert w.sum(axis=1).max() <= 1.0 + EPS


def test_trend_reversion_blend_is_fixed_ratio_of_long_legs(synth):
    """trend_reversion_blend == 0.6·趋势腿 + 0.4·回归腿多头（固定比例，数值核对）。"""
    s = TrendReversionBlendStrategy()
    trend, rev, wt, wr = s.blend_coefficients()
    trend_w = align_weights(DonchianTurtleStrategy().generate_weights(synth), synth,
                            long_only=True).to_numpy()
    rev_w = align_weights(ZscoreReversionStrategy().generate_weights(synth), synth,
                          long_only=True).to_numpy()          # 可多空腿只取多头
    expected = wt * trend_w + wr * rev_w
    w = s.generate_weights(synth).to_numpy()
    assert np.allclose(w, expected, atol=1e-12, rtol=0.0)
    assert (rev_w > 0.0).any() and (trend_w > 0.0).any()
    assert not np.allclose(w, trend_w, atol=1e-6)
    assert not np.allclose(w, rev_w, atol=1e-6)


def test_trend_reversion_blend_drops_short_leg(synth):
    """回归腿的空头被截断：那些格子上元策略权重 == 0.6·趋势腿（long_only 保持）。"""
    s = TrendReversionBlendStrategy()
    raw_rev = ZscoreReversionStrategy().generate_weights(synth)
    trend_w = align_weights(DonchianTurtleStrategy().generate_weights(synth), synth,
                            long_only=True).to_numpy()
    w = s.generate_weights(synth).to_numpy()
    short_cells = (raw_rev.reindex(index=synth.dates, columns=synth.symbols)
                   .fillna(0.0).to_numpy() < 0.0)
    assert short_cells.sum() > 0
    assert np.allclose(w[short_cells], (0.6 * trend_w)[short_cells], atol=1e-12)
    assert w.min() >= 0.0


def test_mix_coefficients_nonneg_and_sum_to_one(synth):
    for cls in WEIGHTED:
        s = cls()
        coef = s.mix_coefficients(synth)
        assert list(coef.columns) == ens.core_base_names(), s.name
        assert coef.index.equals(synth.dates), s.name
        vals = coef.to_numpy()
        assert np.isfinite(vals).all(), s.name
        assert vals.min() >= 0.0, s.name                       # 系数非负
        assert np.allclose(vals.sum(axis=1), 1.0, atol=1e-12), s.name   # 每行和≈1


def test_inverse_vol_coefficients_match_manual_walk_forward(synth):
    s = InverseVolEnsembleStrategy()
    rets = ens.base_backtest_returns(synth)
    expected = _expected_coefficients(rets, "inverse_vol", s.params["window"],
                                      s.params["min_obs"], s.params["vol_floor"],
                                      synth.periods_per_year)
    coef = s.mix_coefficients(synth).to_numpy()
    assert np.allclose(coef, expected, atol=1e-12, rtol=0.0)
    assert (coef[:s.params["min_obs"]] == 1.0 / len(ens.core_base_names())).all()


def test_sharpe_coefficients_match_manual_walk_forward(synth):
    s = SharpeWeightedEnsembleStrategy()
    rets = ens.base_backtest_returns(synth)
    expected = _expected_coefficients(rets, "sharpe", s.params["window"],
                                      s.params["min_obs"], s.params["vol_floor"],
                                      synth.periods_per_year)
    coef = s.mix_coefficients(synth).to_numpy()
    assert np.allclose(coef, expected, atol=1e-12, rtol=0.0)
    assert coef.min() >= 0.0


def test_inverse_vol_prefers_calmest_base_strategy(synth):
    """行为：第 t 期系数最大者 == 截至 t-1 滚动波动最低的基础策略（排名反向）。"""
    s = InverseVolEnsembleStrategy()
    win = int(s.params["window"])
    rets = ens.base_backtest_returns(synth)
    coef = s.mix_coefficients(synth)
    checked = 0
    for t in range(win + 1, N_DAYS, 37):
        sd = rets.iloc[t - win:t].std(ddof=1)          # 只用 [t-win, t-1] 的收益
        assert sd.notna().all()
        row = coef.iloc[t]
        assert row.idxmax() == sd.idxmin(), t
        assert row.idxmin() == sd.idxmax(), t
        assert sd.min() > s.params["vol_floor"], t     # 波动下限未生效，排名纯由 σ 决定
        checked += 1
    assert checked >= 2


def test_sharpe_clips_negative_sharpe_to_zero(synth):
    """行为：截至 t-1 滚动夏普为负的基础策略系数恒为 0（clip>=0），为正的获得权重。"""
    s = SharpeWeightedEnsembleStrategy()
    win, mo, floor = int(s.params["window"]), int(s.params["min_obs"]), float(s.params["vol_floor"])
    rets = ens.base_backtest_returns(synth)
    roll = rets.rolling(win, min_periods=mo)
    sd = roll.std(ddof=1)
    sr = ((roll.mean() / sd.where(sd > floor)) * np.sqrt(synth.periods_per_year)).shift(1)
    coef = s.mix_coefficients(synth).to_numpy()
    sr_vals = sr.to_numpy()
    finite = np.isfinite(sr_vals)
    # 无任何正夏普（含预热期全 NaN）的行回退等权，不参与 clip 断言
    degenerate = (~(finite & (sr_vals > 0.0))).all(axis=1)
    live = ~degenerate
    neg = finite & (sr_vals < 0.0) & live[:, None]
    pos = finite & (sr_vals > 0.0) & live[:, None]
    assert degenerate.sum() >= mo                    # 预热期整段等权回退
    assert neg.sum() > 0 and pos.sum() > 0
    assert (coef[neg] == 0.0).all()                  # 负夏普 -> 系数恰为 0
    assert (coef[pos] > 0.0).all()                   # 正夏普 -> 获得正权重
    k = len(ens.core_base_names())
    assert np.allclose(coef[degenerate], 1.0 / k, atol=1e-12)
    assert np.allclose(coef.sum(axis=1), 1.0, atol=1e-12)


def test_warmup_falls_back_to_equal_weight(synth):
    """预热期（观测 < min_obs）系数等权，元策略权重与 equal_weight_ensemble 完全一致。"""
    eq = EqualWeightEnsembleStrategy().generate_weights(synth).to_numpy()
    k = len(ens.core_base_names())
    for cls in WEIGHTED:
        s = cls()
        mo = int(s.params["min_obs"])
        coef = s.mix_coefficients(synth).to_numpy()
        assert np.allclose(coef[:mo], 1.0 / k, atol=0.0), s.name
        w = s.generate_weights(synth).to_numpy()
        assert np.allclose(w[:mo], eq[:mo], atol=1e-15), s.name
        # 预热期之后确实开始自适应（不再恒等于等权）
        assert not np.allclose(w[mo + 5:], eq[mo + 5:], atol=1e-6), s.name


def test_weighted_ensembles_really_reweight(synth):
    """三个加权/混合元策略的输出两两不同，且都不同于等权元策略。"""
    outs = {cls.name: cls().generate_weights(synth).to_numpy() for cls in STRATEGIES}
    keys = list(outs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.allclose(outs[keys[i]], outs[keys[j]], atol=1e-8), (keys[i], keys[j])


# ------------------------------------------------------------- 防未来函数

def test_no_lookahead_tamper_future_prices(synth):
    """篡改 t0 之后的价格，t0 及之前的元策略权重必须逐元素完全不变。"""
    tampered = _tamper_future(synth, T0)
    assert not np.allclose(tampered.prices.to_numpy()[T0 + 1:],
                           synth.prices.to_numpy()[T0 + 1:])
    for cls in STRATEGIES:
        s = cls()
        w_full = s.generate_weights(synth).to_numpy()
        w_tam = s.generate_weights(tampered).to_numpy()
        assert np.array_equal(w_full[:T0 + 1], w_tam[:T0 + 1]), s.name
        assert not np.array_equal(w_full, w_tam), s.name      # 未来确实被改写了


def test_no_lookahead_coefficients_use_only_past_returns(synth):
    """混合系数的前缀不变性：截断样本重算，前 PREFIX 期系数完全一致。"""
    sub = _truncate(synth, PREFIX)
    rets_sub = ens.base_backtest_returns(sub)
    rets_full = ens.base_backtest_returns(synth)
    assert np.array_equal(rets_full.to_numpy()[:PREFIX], rets_sub.to_numpy())
    for cls in WEIGHTED:
        s = cls()
        full = s.mix_coefficients(synth).to_numpy()[:PREFIX]
        part = s.mix_coefficients(sub).to_numpy()
        assert np.array_equal(full, part), s.name


def test_no_lookahead_prefix_truncation(synth):
    """截断样本重算权重，前缀必须与全样本完全一致（基础腿与元策略都只用历史）。"""
    sub = _truncate(synth, PREFIX)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).to_numpy()[:PREFIX]
        part = s.generate_weights(sub).to_numpy()
        assert np.array_equal(full, part), s.name


def test_no_lookahead_coefficient_ignores_current_period_return(synth):
    """显式验证 shift(1)：只篡改第 t 期（含）之后的收益，第 t 期系数不变。"""
    s = InverseVolEnsembleStrategy()
    win = int(s.params["window"])
    t = win + 60
    rets = ens.base_backtest_returns(synth)
    coef_full = s.mix_coefficients(synth)
    bumped = rets.copy()
    bumped.iloc[t:] = bumped.iloc[t:] + 0.05          # 大幅改写 t 期及之后的策略收益
    coef_part = ens._coefficients(bumped, "inverse_vol", win, int(s.params["min_obs"]),
                                  float(s.params["vol_floor"]), synth.periods_per_year)
    assert np.array_equal(coef_full.iloc[:t + 1].to_numpy(), coef_part.iloc[:t + 1].to_numpy())
    assert not np.array_equal(coef_full.to_numpy(), coef_part.to_numpy())


# ------------------------------------------------------- 性能 / 边界

def test_performance_budget_full_size_universe():
    """1000×8 数据上四个元策略（含 5 次基础回测）应在数秒内完成。"""
    big = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    t0 = time.perf_counter()
    for cls in STRATEGIES:
        w = cls().generate_weights(big)
        assert w.shape == (1000, 8)
        assert w.sum(axis=1).max() <= 1.0 + EPS
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"耗时 {elapsed:.2f}s 超出预算"


def test_tiny_dataset_edge_case():
    """极小数据（5 期 × 1 资产）不崩溃：全程等权回退，约束仍成立。"""
    tiny = MarketData(prices=pd.DataFrame(
        {"A0": np.array([100.0, 101.0, 99.0, 102.0, 103.0])},
        index=pd.bdate_range("2020-01-01", periods=5)))
    for cls in STRATEGIES:
        w = cls().generate_weights(tiny)
        assert w.shape == (5, 1), cls.name
        assert np.isfinite(w.to_numpy()).all(), cls.name
        assert w.to_numpy().min() >= 0.0, cls.name
        assert w.sum(axis=1).max() <= 1.0 + EPS, cls.name
