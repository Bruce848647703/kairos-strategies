import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, discover, make_synthetic_universe
from kairos_strategies.channels import allocation as alloc
from kairos_strategies.channels.allocation import (
    EqualWeightRebalanceStrategy,
    InverseVolatilityStrategy,
    MaxDiversificationStrategy,
    MinVarianceAllocationStrategy,
    RiskParityAllocationStrategy,
)

ALLOC_CLASSES = [
    EqualWeightRebalanceStrategy,
    InverseVolatilityStrategy,
    RiskParityAllocationStrategy,
    MinVarianceAllocationStrategy,
    MaxDiversificationStrategy,
]
# 依赖滚动协方差窗口的策略（预热期应回退等权）
COV_CLASSES = [
    InverseVolatilityStrategy,
    RiskParityAllocationStrategy,
    MinVarianceAllocationStrategy,
    MaxDiversificationStrategy,
]

EPS = 1e-9
WINDOW = 60          # 与协方差类策略默认 params["window"] 一致
REBAL = 21           # EqualWeightRebalanceStrategy 默认 params["rebal_freq"]


def _synth(prices_dict):
    prices = pd.DataFrame(prices_dict)
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    return MarketData(prices=prices)


def _low_high_universe(n=200, seed=0):
    """1 个极低波动资产 + 5 个高波动资产（相互独立）。"""
    rng = np.random.default_rng(seed)
    cols = {"LOW": 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))}
    for i in range(5):
        cols[f"H{i}"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    return _synth(cols)


def _make_cov(vols, corr):
    """等相关结构协方差：Σ = D·C·D，C 为等相关矩阵。"""
    vols = np.asarray(vols, dtype=float)
    N = len(vols)
    C = np.full((N, N), corr, dtype=float)
    np.fill_diagonal(C, 1.0)
    return np.diag(vols) @ C @ np.diag(vols)


# ---------------------------- 通用契约 ----------------------------

@pytest.mark.parametrize("cls", ALLOC_CLASSES)
def test_shape_columns_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == data.symbols
    assert len(w) == len(data.dates)
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", ALLOC_CLASSES)
def test_long_only_weights_in_unit_range(cls):
    data = make_synthetic_universe(n_assets=7, n_days=300, seed=9)
    w = cls().generate_weights(data)
    assert (w.values >= 0.0).all()
    assert (w.values <= 1.0).all()


@pytest.mark.parametrize("cls", ALLOC_CLASSES)
def test_fully_invested_row_sum_approx_one(cls):
    """满仓配置：每行权重和 ≈ 1（含预热期——窗口不足回退等权）。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    row_sum = w.sum(axis=1).values
    assert np.abs(row_sum - 1.0).max() <= EPS
    assert row_sum.max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", COV_CLASSES)
def test_row_sum_one_after_warmup(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert np.abs(w.iloc[WINDOW:].sum(axis=1).values - 1.0).max() <= EPS


@pytest.mark.parametrize("cls", COV_CLASSES)
def test_warmup_falls_back_to_equal_weight(cls):
    """滚动窗口不足（t < window-1）时回退等权 1/N。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    expected = np.full(6, 1.0 / 6.0)
    for t in range(WINDOW - 1):
        np.testing.assert_allclose(w.iloc[t].values, expected, atol=EPS)


@pytest.mark.parametrize("cls", ALLOC_CLASSES)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=250, seed=3)
    a = cls().generate_weights(data)
    b = cls().generate_weights(data)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", ALLOC_CLASSES)
def test_no_arg_construction_and_meta(cls):
    s = cls()                      # 必须能无参构造
    assert isinstance(s.params, dict) and s.params
    m = s.meta()
    assert m["channel"] == "allocation"
    assert m["universe"] == "cross_section"
    assert s.long_only is True
    for k in ("name", "description", "hypothesis", "source", "params"):
        assert m[k]


def test_strategy_names_unique():
    names = [c.name for c in ALLOC_CLASSES]
    assert len(set(names)) == len(names)


def test_discover_finds_allocation_channel():
    names = {s.name for s in discover()}
    assert {"equal_weight_rebal", "inverse_vol", "risk_parity_alloc",
            "min_variance_alloc", "max_diversification"} <= names


# ---------------------------- 行为断言 ----------------------------

def test_inverse_vol_prefers_low_volatility_asset():
    data = _low_high_universe()
    w = InverseVolatilityStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last.idxmax() == "LOW"
    assert last["LOW"] > last.drop("LOW").max()
    assert last["LOW"] > 0.5        # 波动低 ~30 倍 → 权重应占绝对主导


def test_risk_parity_prefers_low_volatility_asset():
    data = _low_high_universe()
    w = RiskParityAllocationStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last.idxmax() == "LOW"
    assert last["LOW"] > last.drop("LOW").max()


def test_risk_parity_equalizes_risk_contributions():
    """求解器级：各资产成分风险贡献 RC_j = w_j(Σw)_j 应相等。"""
    S = alloc._regularize(_make_cov([0.01, 0.02, 0.04, 0.08], 0.3))
    w = alloc._risk_parity_weights(S)
    RC = w * (S @ w)
    assert np.abs(RC / RC.mean() - 1.0).max() < 1e-5
    assert (w > 0).all()
    assert abs(w.sum() - 1.0) < EPS


def test_equal_weight_rebal_resets_and_drifts():
    """再平衡日（t % k == 0）权重回到 1/N；其间随收益漂移（强者权重上升）。"""
    n = 3 * REBAL
    growth = [0.002, 0.001, 0.0]
    cols = {f"A{j}": 100.0 * (1.0 + g) ** np.arange(n)
            for j, g in enumerate(growth)}
    w = EqualWeightRebalanceStrategy().generate_weights(_synth(cols))
    for t in (0, REBAL, 2 * REBAL):
        np.testing.assert_allclose(w.iloc[t].values, np.full(3, 1.0 / 3.0),
                                   atol=1e-12)
    mid = w.iloc[REBAL + 1]
    assert mid["A0"] > 1.0 / 3.0 + 1e-6      # 涨得快的漂移加仓
    assert mid["A2"] < 1.0 / 3.0 - 1e-6      # 不涨的漂移减仓
    assert abs(mid.sum() - 1.0) < EPS


def test_min_variance_leq_equal_weight_on_same_cov():
    """求解器级：同一协方差下，最小方差组合波动 ≤ 等权组合波动。"""
    vols = [0.01, 0.02, 0.04, 0.08]
    for corr in (0.05, 0.3, 0.7):
        S = alloc._regularize(_make_cov(vols, corr))
        w_mv = alloc._min_variance_weights(S)
        w_eq = np.full(len(vols), 1.0 / len(vols))
        vol_mv = np.sqrt(float(w_mv @ S @ w_mv))
        vol_eq = np.sqrt(float(w_eq @ S @ w_eq))
        assert (w_mv >= 0.0).all() and abs(w_mv.sum() - 1.0) < EPS
        assert vol_mv <= vol_eq


def test_min_variance_end_to_end_lower_realized_vol():
    """端到端：低/高波资产组合上，min_variance 的事后已实现波动 ≤ 等权。"""
    data = _low_high_universe()
    w = MinVarianceAllocationStrategy().generate_weights(data)
    rets = data.returns()
    port_mv = (w * rets).sum(axis=1).iloc[WINDOW:]
    port_eq = rets.mean(axis=1).iloc[WINDOW:]
    assert port_mv.std() <= port_eq.std()
    assert w.iloc[-1].idxmax() == "LOW"


def test_max_diversification_matches_inverse_vol_on_diagonal_cov():
    """求解器级：Σ 对角（不相关）时 MDP 解析解为 w ∝ 1/σ。"""
    vols = np.array([0.01, 0.02, 0.04, 0.08])
    S = alloc._regularize(_make_cov(vols, 0.0))
    w = alloc._max_div_weights(S)
    target = (1.0 / vols) / (1.0 / vols).sum()
    np.testing.assert_allclose(w, target, atol=1e-6)


def test_max_diversification_beats_equal_weight_dr():
    """求解器级：MDP 的分散化比率 DR 应不低于等权（含负相关边界解情形）。"""
    vols = [0.01, 0.02, 0.04, 0.08]
    for corr in (0.0, 0.3, 0.7, -0.2):
        S = alloc._regularize(_make_cov(vols, corr))
        sigma = np.sqrt(np.diag(S))

        def _dr(x):
            return float(sigma @ x) / np.sqrt(float(x @ S @ x))

        w = alloc._max_div_weights(S)
        w_eq = np.full(len(vols), 1.0 / len(vols))
        assert (w >= 0.0).all() and abs(w.sum() - 1.0) < EPS
        assert _dr(w) > _dr(w_eq)


def test_max_diversification_prefers_low_volatility_asset():
    data = _low_high_universe()
    w = MaxDiversificationStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last.idxmax() == "LOW"
    assert last["LOW"] > last.drop("LOW").max()
