"""ml_signal 渠道测试。

覆盖：元信息/无参构造、形状/列对齐/有限值、每行绝对值和 ≤ 1+eps、long_only 权重
≥ 0 且活跃行和 ≈ 1、多空行和 ≈ 0（美元中性）且两向持仓、确定性、预热期空仓、
refit 块内权重冻结（持有到下次重训）、依赖白名单（只用 numpy/pandas，禁止
sklearn/statsmodels/scipy）、退化 universe 边界、性能冒烟（8 资产 × 1000 天）。

**防未来函数**（关键断言）：
  * 篡改 t 之后的价格 → t 及之前的权重逐位不变（np.array_equal）；
  * 更强：本渠道第 t 行权重只依赖 ≤ t−1 的价格，连 t 当期价格一起篡改也不变；
  * 截断样本重算 → 前缀权重逐位不变（refit 网格锚定绝对索引 0）。

**行为断言**（模型确实学到信号，优于随机）：构造「特征与下期收益线性相关」的
合成场景 —— A0 的收益为强自相关 AR(1)（r_{s+1} = φ·r_s + 小噪声，φ=0.8），其余
资产纯噪声，末段再给 A0 一段确定性稳定上坡 —— 验证 ridge_alpha / gbm_alpha
（以及 logit_signal / knn_alpha）在最后一个 refit 块给 A0 **严格行内最大**权重，
且超过等权份额 1/N（随机配置的期望水平）。
"""
from __future__ import annotations

import ast
import inspect
import time

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.ml_signal import (
    GbmAlphaStrategy,
    KnnAlphaStrategy,
    LogitSignalStrategy,
    RidgeAlphaStrategy,
)

STRATEGIES = [RidgeAlphaStrategy, LogitSignalStrategy, GbmAlphaStrategy, KnnAlphaStrategy]
EXPECTED_NAMES = {"ridge_alpha", "logit_signal", "gbm_alpha", "knn_alpha"}
LONG_ONLY = [RidgeAlphaStrategy, LogitSignalStrategy]
LONG_SHORT = [GbmAlphaStrategy, KnnAlphaStrategy]
EPS = 1e-9          # 绝对值和 / 值域容差
SUM_EPS = 1e-9      # 美元中性行和容差


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _signal_universe(n: int = 420, n_assets: int = 6, phi: float = 0.80,
                     sig_noise: float = 0.005, other_noise: float = 0.004,
                     rally: int = 45, per_day: float = 0.012,
                     seed: int = 3) -> MarketData:
    """「特征与下期收益线性相关」的合成场景。

    A0 的收益是强自相关 AR(1)：``r_{s+1} = φ·r_s + 小噪声``（φ=0.8），即滞后收益
    特征对其下期收益有清晰的**线性**预测力；其余资产是与其无关的纯噪声。末段
    ``rally`` 期把 A0 的收益覆写为每期 +``per_day`` 的确定性上坡，使最后一个
    refit 日上 A0 的动量/RSI/均线偏离等特征在截面中极端偏高 —— 学到信号的模型
    必须给 A0 最高预测与最高权重。
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, n_assets))
    r = np.zeros((n, n_assets))
    r[0, 0] = eps[0, 0] * sig_noise
    for t in range(1, n):
        r[t, 0] = phi * r[t - 1, 0] + eps[t, 0] * sig_noise
    r[:, 1:] = eps[:, 1:] * other_noise
    if rally > 0:
        r[n - rally:, 0] = per_day
    prices = 100.0 * np.exp(np.cumsum(r, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    return MarketData(prices=pd.DataFrame(prices, index=_bdays(n), columns=cols))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=420, seed=11)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "ml_signal"
        assert s.universe == "cross_section"
        assert s.long_only is (cls in LONG_ONLY)
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 10, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"]
        assert int(meta["params"]["refit"]) >= 1
        assert int(meta["params"]["min_train_dates"]) >= 2
    assert names == EXPECTED_NAMES


def test_shape_columns_index_alignment(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == (len(synth.dates), len(synth.symbols)), cls.name
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
        assert w.values.min() >= -1.0 - EPS, cls.name
        assert w.values.max() <= 1.0 + EPS, cls.name


def test_long_only_constraints(synth):
    for cls in LONG_ONLY:
        w = cls().generate_weights(synth)
        assert w.values.min() >= 0.0, cls.name               # 权重非负
        row_sum = w.sum(axis=1)
        assert (row_sum <= 1.0 + EPS).all(), cls.name        # 每行和 ≤ 1
        active = row_sum > 0
        assert active.any(), cls.name
        assert np.allclose(row_sum[active].values, 1.0, atol=1e-9), cls.name


def test_long_short_constraints(synth):
    for cls in LONG_SHORT:
        w = cls().generate_weights(synth)
        row_sum = w.sum(axis=1)
        assert np.abs(row_sum.values).max() <= SUM_EPS, cls.name   # 美元中性
        abs_sum = w.abs().sum(axis=1)
        active = abs_sum > 0
        assert int(active.sum()) >= 50, (cls.name, int(active.sum()))
        assert np.allclose(abs_sum[active].values, 1.0, atol=1e-9), cls.name
        assert w.values.min() < 0.0, f"{cls.name} 从未做空"
        assert w.values.max() > 0.0, f"{cls.name} 从未做多"


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        # 不同实例（同参数）也必须一致：策略无可变状态、无随机数
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)


def test_warmup_flat_and_block_hold(synth):
    """预热期整行空仓；refit 网格锚定绝对索引；块内权重冻结（持有到下次 refit）。"""
    T = len(synth.dates)
    for cls in STRATEGIES:
        s = cls()
        w = s.generate_weights(synth)
        t0 = s.first_signal_index()
        assert (w.iloc[:t0].values == 0.0).all(), cls.name   # 预热空仓
        refit = int(s.params["refit"])
        blocks = list(range(t0, T, refit))
        assert blocks, cls.name
        for t_k in blocks:
            blk = w.iloc[t_k:min(t_k + refit, T)].values
            assert np.abs(blk).sum() > 0.0, (cls.name, t_k)  # 合成数据下每块都应持仓
            assert (blk == blk[0]).all(), (cls.name, t_k)    # 块内逐位相同（冻结）


# ------------------------------------------------------------------ 防未来函数

def test_no_lookahead_tamper_future(synth):
    """篡改 t 之后的价格（缩放 + 逆序 + 斜坡，剧烈改动），t 及之前的权重逐位不变。

    更强断言：本渠道第 τ 行权重只依赖 ≤ τ−1 的价格（训练标签最晚 r_{t_k−1}、
    预测特征在 t_k−1，块内冻结），因此连 t 当期价格一起篡改，≤ t 的权重仍逐位不变。
    """
    T = len(synth.dates)
    N = len(synth.symbols)
    cut = 260
    p0 = synth.prices

    p_tam = p0.copy()
    tail = p_tam.iloc[cut + 1:].to_numpy()
    ramp = np.linspace(0.5, 3.0, tail.shape[0])[:, None]
    fac = np.array([1.7 + 0.37 * (i % 5) for i in range(N)]).reshape(1, -1)
    p_tam.iloc[cut + 1:] = tail[::-1] * ramp * fac           # 剧烈篡改，恒为正
    d_tam = MarketData(prices=p_tam, volumes=None,
                       periods_per_year=synth.periods_per_year, name=synth.name)

    p_tam2 = p0.copy()
    p_tam2.iloc[cut:] = p_tam2.iloc[cut:] * 2.0              # 连 t 当期一起篡改
    d_tam2 = MarketData(prices=p_tam2, volumes=None,
                        periods_per_year=synth.periods_per_year, name=synth.name)

    assert cut + 1 <= T
    for cls in STRATEGIES:
        s = cls()
        w_full = s.generate_weights(synth).values
        w_tam = s.generate_weights(d_tam).values
        w_tam2 = s.generate_weights(d_tam2).values
        assert np.array_equal(w_full[:cut + 1], w_tam[:cut + 1]), cls.name
        assert np.array_equal(w_full[:cut + 1], w_tam2[:cut + 1]), cls.name


def test_no_lookahead_truncation_prefix(synth):
    """截断样本重算：前缀权重逐位不变（refit 网格与训练窗口都锚定绝对索引）。"""
    m = 320
    sub = MarketData(prices=synth.prices.iloc[:m], volumes=None,
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.array_equal(full, part), cls.name


# ------------------------------------------------------------------ 依赖与边界

def test_only_numpy_pandas_dependencies():
    """禁止 sklearn / statsmodels / scipy：四个模型必须是自研 numpy 实现。"""
    import kairos_strategies.channels.ml_signal as mod
    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "pandas", "typing", "__future__", "abc"}, imported
    for bad in ("sklearn", "statsmodels", "scipy"):
        assert bad not in mod.__dict__, bad


def test_degenerate_universes_are_safe():
    """单资产 / 历史过短 / 两资产 / 恒定价格 等退化输入都必须安全返回。"""
    base = make_synthetic_universe(n_assets=3, n_days=400, seed=5)
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    flat = MarketData(prices=pd.DataFrame(100.0, index=base.prices.index,
                                          columns=base.prices.columns))
    for cls in STRATEGIES:
        s = cls()
        for d in (one, short, two, flat):
            w = s.generate_weights(d)
            assert w.shape == (len(d.dates), len(d.symbols)), cls.name
            assert list(w.columns) == d.symbols
            assert np.isfinite(w.values).all(), cls.name
            assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
            if cls.long_only:
                assert w.values.min() >= 0.0, cls.name
            else:
                assert np.abs(w.sum(axis=1).values).max() <= SUM_EPS, cls.name
        assert (s.generate_weights(short).values == 0.0).all()   # 历史过短全空仓
        # 单资产做多空凑不出两腿 → 全 0；只做多则最多满仓该资产
        w_one = s.generate_weights(one)
        if not cls.long_only:
            assert (w_one.values == 0.0).all(), cls.name


# ------------------------------------------------------------------ 行为断言

def _last_block(s, T: int):
    t0 = s.first_signal_index()
    refit = int(s.params["refit"])
    lo = t0 + ((T - 1 - t0) // refit) * refit
    return lo, min(lo + refit, T)


@pytest.mark.parametrize("cls", [RidgeAlphaStrategy, GbmAlphaStrategy,
                                 LogitSignalStrategy, KnnAlphaStrategy])
def test_model_learns_linear_signal(cls):
    """行为断言：A0 的特征与其下期收益线性相关（AR(1) φ=0.8）+ 末段确定性上坡。

    最后一个 refit 块内，模型必须给 A0 **严格行内最大**权重（做多信号资产），
    且超过等权份额 1/N —— 即明显优于随机配置，证明模型确实学到了植入的信号。
    重点是 ridge_alpha（线性闭式解）与 gbm_alpha（非线性集成）两者。
    """
    d = _signal_universe()
    s = cls()
    lo, hi = _last_block(s, len(d.dates))
    w = s.generate_weights(d)
    blk = w.iloc[lo:hi]
    assert blk.shape[0] > 0 and np.abs(blk.to_numpy()).sum() > 0.0, "末块不应空仓"
    a0 = blk["A0"].to_numpy()
    others = blk.drop(columns=["A0"]).to_numpy()
    assert (a0 == blk.to_numpy().max(axis=1)).all(), cls.name      # A0 是行内最大
    assert (a0 > others.max(axis=1)).all(), cls.name               # 严格大于其余资产
    assert (a0 > 0.0).all(), cls.name                              # 做多信号资产
    assert (a0 > 1.0 / d.n_assets).all(), cls.name                 # 优于随机/等权


def test_ridge_alpha_signal_stronger_than_noise_assets():
    """补充行为断言：ridge_alpha 对信号资产的配置远高于噪声资产的总配置。"""
    d = _signal_universe()
    s = RidgeAlphaStrategy()
    lo, hi = _last_block(s, len(d.dates))
    w = s.generate_weights(d).iloc[lo:hi]
    a0 = w["A0"].to_numpy()
    rest = w.drop(columns=["A0"]).to_numpy().sum(axis=1)
    assert (a0 > rest).all()
    assert float(a0.mean()) > 0.5                                  # 组合一半以上押注信号资产


# ------------------------------------------------------------------ 性能冒烟

def test_performance_smoke():
    """n_assets=8、n_days=1000：四个策略全部跑完应远小于 60s（实际秒级）。"""
    d = make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    t0 = time.monotonic()
    for cls in STRATEGIES:
        w = cls().generate_weights(d)
        assert w.shape == (1000, 8), cls.name
        assert np.isfinite(w.values).all(), cls.name
    elapsed = time.monotonic() - t0
    assert elapsed < 60.0, f"四策略总耗时 {elapsed:.1f}s，超出预算"
