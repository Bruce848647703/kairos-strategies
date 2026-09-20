import sys

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels import hrp
from kairos_strategies.channels.hrp import (
    ClusteredInverseVolatilityStrategy,
    HierarchicalEqualRiskContributionStrategy,
    HierarchicalRiskParityStrategy,
)

HRP_CLASSES = [
    HierarchicalRiskParityStrategy,
    HierarchicalEqualRiskContributionStrategy,
    ClusteredInverseVolatilityStrategy,
]

EPS = 1e-9
WINDOW = 60          # 与各策略默认 params["window"] 一致
SCIPY_MODS = ("scipy", "scipy.cluster", "scipy.cluster.hierarchy",
              "scipy.spatial", "scipy.spatial.distance")

# 明确的 4 资产距离矩阵（single-linkage 无并列）：
# 先并 (0,1)@0.10，再并 (2,3)@0.20，最后 (4,5)@0.60
D4 = np.array([
    [0.00, 0.10, 0.60, 0.70],
    [0.10, 0.00, 0.65, 0.75],
    [0.60, 0.65, 0.00, 0.20],
    [0.70, 0.75, 0.20, 0.00],
])
MERGES4 = [(0, 1), (2, 3), (4, 5)]


def _synth(prices_dict):
    prices = pd.DataFrame(prices_dict)
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    return MarketData(prices=prices)


def _two_cluster_universe(n_days=320, seed=7):
    """两组明显相关簇：簇内相关 ~0.98（共享因子），簇间相关 ~0。"""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0, 0.010, n_days)
    f2 = rng.normal(0.0, 0.010, n_days)
    cols = {}
    for j in range(3):
        cols[f"A{j}"] = 100 * np.exp(np.cumsum(f1 + rng.normal(0, 0.0015, n_days)))
    for j in range(3):
        cols[f"B{j}"] = 100 * np.exp(np.cumsum(f2 + rng.normal(0, 0.0015, n_days)))
    return _synth(cols)


def _vol_asym_universe(n_days=320, seed=11):
    """簇1={LOW, HIGH}：同一因子、波动 1:3（高相关）；簇2={X, Y}：低波等权对照。"""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0, 0.020, n_days)
    f2 = rng.normal(0.0, 0.010, n_days)
    low = f1 + rng.normal(0, 0.002, n_days)            # sd ≈ 0.020
    high = 3.0 * f1 + rng.normal(0, 0.002, n_days)     # sd ≈ 0.060，与 low 相关 ≈ 0.99
    x = f2 + rng.normal(0, 0.001, n_days)
    y = f2 + rng.normal(0, 0.001, n_days)
    cols = {k: 100 * np.exp(np.cumsum(r))
            for k, r in [("LOW", low), ("HIGH", high), ("X", x), ("Y", y)]}
    return _synth(cols)


def _block_scipy(monkeypatch):
    """把 sys.modules 里的 scipy 及其子模块置 None，使运行时 import 失败。"""
    for name in SCIPY_MODS:
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------- 通用契约 ----------------------------

@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_shape_columns_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == data.symbols
    assert len(w) == len(data.dates)
    assert np.isfinite(w.values).all()


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_long_only_weights_in_unit_range(cls):
    data = make_synthetic_universe(n_assets=7, n_days=300, seed=9)
    w = cls().generate_weights(data)
    assert (w.values >= 0.0).all()
    assert (w.values <= 1.0).all()


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_fully_invested_row_sum_approx_one(cls):
    """满仓配置：每行权重和 ≈ 1（含预热期——窗口不足回退等权），且 ≤ 1+eps。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    row_sum = w.sum(axis=1).values
    assert np.abs(row_sum - 1.0).max() <= EPS
    assert row_sum.max() <= 1.0 + EPS


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_determinism_two_runs_identical(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w1 = cls().generate_weights(data)
    w2 = cls().generate_weights(data)
    assert np.array_equal(w1.values, w2.values)


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_warmup_falls_back_to_equal_weight(cls):
    """滚动窗口不足（t < window-1）时回退等权 1/N。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    warm = w.iloc[:WINDOW - 1].values
    assert np.allclose(warm, 1.0 / data.n_assets, atol=EPS)


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_no_lookahead_truncated_history_matches(cls):
    """防未来：把历史截断到前 m 天，前 m 行权重必须与全样本一致。"""
    data = make_synthetic_universe(n_assets=5, n_days=250, seed=13)
    m = 180
    w_full = cls().generate_weights(data)
    trunc = MarketData(prices=data.prices.iloc[:m])
    w_trunc = cls().generate_weights(trunc)
    assert np.abs(w_full.values[:m] - w_trunc.values).max() <= EPS


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_degenerate_constant_prices(cls):
    """退化输入（一列恒定价格 → 零方差）仍输出合法权重：有限、≥0、和≈1。"""
    n = 120
    cols = {"FLAT": np.full(n, 100.0),
            "MOVE": 100 * np.exp(np.cumsum(np.linspace(-0.002, 0.002, n)))}
    data = _synth(cols)
    w = cls().generate_weights(data)
    assert np.isfinite(w.values).all()
    assert (w.values >= 0.0).all()
    assert np.abs(w.sum(axis=1).values - 1.0).max() <= EPS


def test_single_asset_all_weights_one():
    n = 80
    cols = {"ONLY": 100 * np.exp(np.cumsum(np.full(n, 0.001)))}
    data = _synth(cols)
    for cls in HRP_CLASSES:
        w = cls().generate_weights(data)
        assert np.allclose(w.values, 1.0, atol=EPS)


# ---------------------------- 元信息 ----------------------------

def test_meta_complete_and_names_unique():
    names = set()
    for cls in HRP_CLASSES:
        s = cls()                      # 无参可构造
        m = s.meta()
        assert m["channel"] == "hrp"
        assert m["universe"] == "cross_section"
        assert m["long_only"] is True
        assert m["name"] and isinstance(m["params"], dict)
        assert m["description"] and m["hypothesis"] and m["source"]
        names.add(m["name"])
    assert names == {"hrp", "herc", "clustered_inverse_vol"}


# ---------------------------- 聚类内核 ----------------------------

def test_agglomerative_fallback_merges():
    """numpy 自研凝聚聚类（single-linkage）在明确距离下合并序列正确。"""
    assert hrp._agglomerative_merges(D4) == MERGES4


def test_linkage_merges_scipy_path():
    """scipy 可用时 _linkage_merges 与 numpy 回退给出相同的 single-linkage 序列。"""
    pytest.importorskip("scipy")
    assert hrp._scipy_linkage(D4) is not None
    assert hrp._linkage_merges(D4) == MERGES4


def test_linkage_merges_fallback_when_scipy_import_fails(monkeypatch):
    """scipy import 失败（sys.modules 置 None）时自动回退且结果一致。"""
    _block_scipy(monkeypatch)
    assert hrp._scipy_linkage(D4) is None
    assert hrp._linkage_merges(D4) == MERGES4


def test_flat_clusters_partition():
    clusters = hrp._flat_clusters(MERGES4, 4, 2)
    assert clusters == [[0, 1], [2, 3]]
    assert hrp._flat_clusters(MERGES4, 4, 4) == [[0], [1], [2], [3]]
    assert hrp._flat_clusters(MERGES4, 4, 1) == [[0, 1, 2, 3]]


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_weights_valid_without_scipy(cls, monkeypatch):
    """scipy 缺失路径（numpy 回退聚类）仍给出合法权重：有限、≥0、和≈1。"""
    _block_scipy(monkeypatch)
    data = make_synthetic_universe(n_assets=6, n_days=160, seed=3)
    w = cls().generate_weights(data)
    assert np.isfinite(w.values).all()
    assert (w.values >= 0.0).all()
    assert np.abs(w.sum(axis=1).values - 1.0).max() <= EPS


def test_hrp_weights_unit_simplex_direct():
    """_hrp_weights 直接输出 long-only 且和=1（3 资产明确相关结构）。"""
    S = np.array([[4.0, 3.0, 0.0],
                  [3.0, 9.0, 0.0],
                  [0.0, 0.0, 1.0]]) * 1e-4
    merges = hrp._agglomerative_merges(
        hrp._distance_from_corr(hrp._corr_from_cov(hrp._regularize(S))))
    w = hrp._hrp_weights(hrp._regularize(S), merges)
    assert (w >= 0.0).all()
    assert abs(w.sum() - 1.0) <= EPS


# ---------------------------- 行为断言 ----------------------------

def test_two_clusters_get_roughly_equal_total_weight():
    """行为：两组明显相关簇（簇内高相关、簇间低相关）获得大致相当的总权重。

    - clustered_inverse_vol / herc（k=2）：簇间等权 → 每簇恰为 0.5；
    - hrp：递归二分按簇方差倒数分配，两簇结构对称 → 份额在 [0.3, 0.7]，
      不会把资金全压在一个簇上。
    """
    data = _two_cluster_universe(n_days=320, seed=7)
    a_cols = ["A0", "A1", "A2"]

    for cls in (ClusteredInverseVolatilityStrategy,
                HierarchicalEqualRiskContributionStrategy):
        w = cls().generate_weights(data).iloc[WINDOW:]
        share_a = w[a_cols].sum(axis=1).values
        assert np.abs(share_a - 0.5).max() <= EPS

    w = HierarchicalRiskParityStrategy().generate_weights(data).iloc[WINDOW:]
    share_a = w[a_cols].sum(axis=1).values
    assert share_a.min() >= 0.30 and share_a.max() <= 0.70


@pytest.mark.parametrize("cls", HRP_CLASSES)
def test_high_vol_asset_gets_lower_weight(cls):
    """行为：同簇内高波动资产权重更低（LOW:HIGH 波动 ≈ 1:3，相关 ≈ 0.99）。"""
    data = _vol_asym_universe(n_days=320, seed=11)
    w = cls().generate_weights(data).iloc[WINDOW - 1:]
    assert (w["LOW"].values > w["HIGH"].values).all()


def test_clustered_inverse_vol_ratio_matches_vol_ratio():
    """行为：clustered_inverse_vol 簇内权重比 ≈ 波动反比（HIGH/LOW ≈ 3 ⇒ ≈ 3）。"""
    data = _vol_asym_universe(n_days=320, seed=11)
    w = ClusteredInverseVolatilityStrategy().generate_weights(data)
    ratio = (w["LOW"] / w["HIGH"]).iloc[WINDOW:].values
    assert np.abs(ratio - 3.0).max() < 0.5
