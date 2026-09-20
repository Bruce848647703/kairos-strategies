"""multi_factor 渠道测试。

覆盖：
  * 契约：4 个策略名精确且互不重复、channel/universe/long_only 正确、元信息齐全
    （中文 description/hypothesis/source）、无参可构造、params 完整；
  * 数值健全：形状 / index / columns 对齐、全有限、权重 ∈ [0,1]、每行和 ≈ 1
    （预热行为 0，永不超过 1+eps）、篮子规模 = ceil(top_frac·N)、确定性；
  * 预热：``first_active_index()`` 之前整行空仓、当期立即满仓（由参数推导）；
  * **防未来函数**：篡改第 m 期及之后的价格后，第 ≤ m 期的权重 / 复合分 / 因子复合
    权重面板（尤其 ic_weighted、max_ir 的滚动 IC 统计与 pca 的特征分解载荷）以及
    基础因子面板全部**逐位（bitwise）不变**；截断样本重算的前缀与全样本逐位一致；
  * 工具函数单元核对：截面 z、秩相关 IC、ic_weight_panel、pca_weight_panel、
    top_basket_weights；
  * **行为断言**：
      - equalweight_composite / pca_factor：动量、低波动、趋势质量三维同时占优的
        STAR 资产拿到最高权重（且每个交易日都是篮子第一名）；
      - ic_weighted_composite：趋势市里历史 IC 最高的动量因子拿到最大权重、历史 IC
        为负的短期反转因子拿到负权，且**逐行**权重排序与 IC 均值排序一致；
      - max_ir_composite：逐行权重排序与 IR = mean(IC)/std(IC) 排序一致，且「IC 略低
        但稳定」的因子权重高于「IC 更高但抖动大」的因子；
  * 依赖白名单（只用 numpy/pandas）、discover() 全局注册与命名唯一、引擎可回测。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.multi_factor import (
    FACTOR_ORDER,
    EqualWeightCompositeStrategy,
    IcWeightedCompositeStrategy,
    MaxIrCompositeStrategy,
    PcaFactorStrategy,
    base_factor_panels,
    cross_sectional_z,
    factor_ic_series,
    ic_series_from_panels,
    ic_weight_panel,
    lagged_ic_stats,
    pca_weight_panel,
    top_basket_weights,
    xs_normalize_stack,
)

STRATEGIES = [EqualWeightCompositeStrategy, IcWeightedCompositeStrategy,
              MaxIrCompositeStrategy, PcaFactorStrategy]
IC_STRATEGIES = [IcWeightedCompositeStrategy, MaxIrCompositeStrategy]
EXPECTED_NAMES = {"equalweight_composite", "ic_weighted_composite",
                  "max_ir_composite", "pca_factor"}
EPS = 1e-9            # 值域 / 绝对值和容差
SUM_EPS = 1e-9        # 每行权重和（≈1）容差
N_FACTORS = len(FACTOR_ORDER)


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _star_universe(n: int = 320, seed: int = 17) -> MarketData:
    """STAR 平滑强势上涨（高动量 + 极低波动 + 效率比≈1）；CHOPPY 大幅震荡；4 只随机游走。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {
        "STAR": 100.0 * np.exp(np.linspace(0.0, 1.6, n)
                               + np.cumsum(rng.standard_normal(n) * 0.0004)),
        "CHOPPY": 100.0 * (1.0 + 0.16 * np.sin(t * 1.1))
                          * np.exp(np.cumsum(rng.standard_normal(n) * 0.02)),
    }
    for i in range(4):
        cols[f"M{i}"] = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.012))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2019-01-01")))


def _drift_universe(n: int = 460, seed: int = 23) -> MarketData:
    """6 只资产漂移阶梯分明（+0.55 → −0.55）+ 小噪声：动量 IC 显著为正、短期反转 IC 为负。"""
    rng = np.random.default_rng(seed)
    drifts = {"T2": 0.55, "T1": 0.30, "F0": 0.05, "F1": -0.05, "D1": -0.30, "D2": -0.55}
    cols = {k: 100.0 * np.exp(np.linspace(0.0, v, n)
                              + np.cumsum(rng.standard_normal(n) * 0.004))
            for k, v in drifts.items()}
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2015-01-01")))


def _factor_universe(n: int = 240, seed: int = 5) -> MarketData:
    """逐因子定向数据：UP 平滑上涨、DOWN 平滑下跌、QUIET 几乎不动、WILD 高波、DROP 近端暴跌。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    quiet = np.where(t % 2 == 0, 1.0002, 0.9998)                 # 微小交替抖动：波动≈0、效率比≈0
    cols = {
        "UP": 100.0 * np.exp(np.linspace(0.0, 0.8, n) + np.cumsum(rng.standard_normal(n) * 0.0005)),
        "DOWN": 100.0 * np.exp(np.linspace(0.0, -0.8, n) + np.cumsum(rng.standard_normal(n) * 0.0005)),
        "QUIET": 100.0 * quiet,
        "WILD": 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.05)),
    }
    flat = np.full(n - 5, 100.0)
    cols["DROP"] = np.concatenate([flat, np.linspace(100.0, 70.0, 6)[1:]])   # 最近 5 天暴跌 30%
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2020-01-01")))


def _tamper_tail(data: MarketData, m: int, seed: int = 7) -> MarketData:
    """把第 m 期（含）之后的价格整体换成一条完全不同的随机路径（仍恒正）。"""
    rng = np.random.default_rng(seed)
    p2 = data.prices.copy()
    shape = p2.iloc[m:].shape
    p2.iloc[m:] = (p2.iloc[m - 1].to_numpy(dtype="float64")[None, :]
                   * np.exp(np.cumsum(rng.standard_normal(shape) * 0.04, axis=0)))
    assert (p2.iloc[m:].to_numpy() != data.prices.iloc[m:].to_numpy()).any()
    return MarketData(prices=p2, volumes=data.volumes,
                      periods_per_year=data.periods_per_year, name=data.name)


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=520, seed=11)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "multi_factor"
        assert s.universe == "cross_section"
        assert s.long_only is True
        assert s.name in EXPECTED_NAMES and s.name not in names, s.name
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"]
        for key in ("mom_window", "vol_window", "rev_window", "qual_window", "top_frac"):
            assert key in meta["params"], (s.name, key)
    assert names == EXPECTED_NAMES
    assert N_FACTORS == 4                                    # 动量/低波/反转/趋势质量


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
        assert not w.isna().any().any(), cls.name


def test_long_only_range_and_row_sum(synth):
    """只做多硬约束：权重 ∈ [0,1]，每行和 ≈ 1（预热行为 0），绝不超过 1+eps。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() >= -EPS, cls.name
        assert w.values.max() <= 1.0 + EPS, cls.name
        row_sum = w.sum(axis=1)
        assert row_sum.max() <= 1.0 + EPS, (cls.name, row_sum.max())
        assert np.all(np.isclose(row_sum.values, 0.0, atol=1e-12)
                      | np.isclose(row_sum.values, 1.0, atol=SUM_EPS)), cls.name
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name        # 不加杠杆


def test_fully_invested_after_warmup(synth):
    """预热结束后每一行都必须满仓（和 ≈ 1），篮子规模 = ceil(top_frac·N)。"""
    n = len(synth.symbols)
    for cls in STRATEGIES:
        s = cls()
        w = s.generate_weights(synth)
        t0 = s.first_active_index()
        act = w.iloc[t0:]
        assert len(act) >= 200, (cls.name, len(act))
        assert np.allclose(act.sum(axis=1).values, 1.0, atol=SUM_EPS), cls.name
        k = int(np.ceil(float(s.params["top_frac"]) * n - 1e-9))
        per_row = np.unique((act.values > 0.0).sum(axis=1))
        assert per_row.tolist() == [k], (cls.name, per_row, k)


def test_warmup_rows_are_flat(synth):
    """信息不足的预热期必须整行空仓；预热长度由参数推导且与样本长度无关。"""
    for cls in STRATEGIES:
        s = cls()
        t0 = s.first_active_index()
        w = s.generate_weights(synth)
        assert (w.iloc[:t0].values == 0.0).all(), (cls.name, t0)
        assert w.iloc[t0].sum() > 0.0, (cls.name, t0)
        assert t0 > max(int(s.params["mom_window"]), int(s.params["vol_window"]),
                        int(s.params["rev_window"]), int(s.params["qual_window"]))
        if s.weighting in ("ic", "ir"):                       # IC 统计需要额外历史
            assert t0 > EqualWeightCompositeStrategy().first_active_index(), cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)  # 无可变状态
        # 复合分与因子权重面板同样确定（PCA 特征分解 / IC 滚动统计无随机性）
        assert np.array_equal(s.composite_scores(synth).values,
                              cls().composite_scores(synth).values, equal_nan=True)
        assert np.array_equal(s.factor_weights(synth).values,
                              cls().factor_weights(synth).values, equal_nan=True)


# ------------------------------------------------------------------ 防未来函数

def test_no_lookahead_tamper_future_prices(synth):
    """篡改第 m 期（含）之后的全部价格：第 ≤ m 期的权重逐位不变。

    同时核对复合分与**因子复合权重面板**（ic_weighted / max_ir 的滚动 IC 统计、
    pca 的特征分解载荷）——即权重计算本身也只吃 ≤ t−1 的已实现数据。
    """
    m = 300
    tam = _tamper_tail(synth, m)
    for cls in STRATEGIES:
        s = cls()
        w0 = s.generate_weights(synth).values
        w1 = s.generate_weights(tam).values
        assert np.array_equal(w0[:m + 1], w1[:m + 1]), cls.name          # 逐位（bitwise）
        sc0 = s.composite_scores(synth).values
        sc1 = s.composite_scores(tam).values
        assert np.array_equal(sc0[:m + 1], sc1[:m + 1], equal_nan=True), cls.name
        fw0 = s.factor_weights(synth).values
        fw1 = s.factor_weights(tam).values
        assert np.array_equal(fw0[:m + 1], fw1[:m + 1], equal_nan=True), cls.name
        fac0, fac1 = s.factor_panels(synth), s.factor_panels(tam)
        for k in FACTOR_ORDER:
            # 因子面板只用「截至当期」价格 -> ≤ m−1 期逐位不变（第 m 期因子含被篡改的 p_m）
            assert np.array_equal(fac0[k].values[:m], fac1[k].values[:m],
                                  equal_nan=True), (cls.name, k)
        assert not np.array_equal(w0[m + 1:], w1[m + 1:]), cls.name      # 篡改确实生效


def test_no_lookahead_prefix_invariance(synth):
    """截断样本重算：前缀权重 / 复合分 / 因子权重面板与全样本逐位一致。"""
    m = 400
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        assert np.array_equal(s.generate_weights(synth).values[:m],
                              s.generate_weights(sub).values), cls.name
        assert np.array_equal(s.composite_scores(synth).values[:m],
                              s.composite_scores(sub).values, equal_nan=True), cls.name
        assert np.array_equal(s.factor_weights(synth).values[:m],
                              s.factor_weights(sub).values, equal_nan=True), cls.name


def test_ic_stats_use_only_realized_labels(synth):
    """IC 统计量的标签滞后：第 t 行的 IC 均值只由 ≤ t−2 期的 IC（标签 ≤ r_{t−1}）构成。

    独立复算：把 ``factor_ic_series`` 的单期 IC 面板手动 shift(2) 后做同样的尾部滚动，
    结果必须与策略内部使用的统计量逐位一致；再验证「只改最后两期价格」不会改动
    倒数第 3 期及之前的任何 IC 统计（标签尚未实现的部分被严格隔离）。
    """
    for cls in IC_STRATEGIES:
        s = cls()
        ic = factor_ic_series(synth, s.params)
        assert list(ic.columns) == list(FACTOR_ORDER)
        lag = int(s.params["label_lag"])
        win = int(s.params["ic_window"])
        mp = int(s.params["ic_min_obs"])
        ref_mean = ic.shift(lag).rolling(win, min_periods=mp).mean()
        ref_std = ic.shift(lag).rolling(win, min_periods=mp).std()
        got_mean, got_std = s.ic_stats(synth)
        pd.testing.assert_frame_equal(got_mean, ref_mean)
        pd.testing.assert_frame_equal(got_std, ref_std)
        # 单期 IC 的标签是「下一期收益」：最后一期没有标签 -> NaN
        assert ic.iloc[-1].isna().all(), cls.name
        assert np.nanmax(np.abs(ic.values)) <= 1.0 + EPS, cls.name
        # 篡改最后两期价格：倒数第 3 期（含）之前的 IC 统计逐位不变
        T = len(synth.dates)
        tam = _tamper_tail(synth, T - 2)
        m0, _ = s.ic_stats(tam)
        cut = T - 2
        assert np.array_equal(got_mean.values[:cut], m0.values[:cut], equal_nan=True), cls.name


def test_ic_series_is_spearman_of_next_period_return():
    """秩相关 IC 的语义核对：因子 = 下一期收益 -> IC ≈ +1；因子 = −下一期收益 -> IC ≈ −1。"""
    n, n_asset = 120, 5
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.standard_normal((n, n_asset)) * 0.01, axis=0)),
        index=_bdays(n), columns=[f"S{i}" for i in range(n_asset)])
    fwd = prices.shift(-1) / prices - 1.0
    ic_pos = ic_series_from_panels([fwd], fwd, min_assets=3)[:, 0]
    ic_neg = ic_series_from_panels([-fwd], fwd, min_assets=3)[:, 0]
    inner = np.isfinite(ic_pos)
    assert inner.sum() >= n - 5
    assert np.allclose(ic_pos[inner], 1.0, atol=1e-12)
    assert np.allclose(ic_neg[inner], -1.0, atol=1e-12)
    assert not np.isfinite(ic_pos[-1])                        # 末期无未来收益 -> 无标签


# ------------------------------------------------------------------ 基础因子与工具函数

def test_base_factors_semantics_and_causality():
    """四个基础因子的方向语义（只用价格/波动）+ 篡改未来价格后因子面板逐位不变。"""
    data = _factor_universe()
    params = EqualWeightCompositeStrategy().params
    fac = base_factor_panels(data, params)
    assert list(fac) == list(FACTOR_ORDER)
    last = {k: v.iloc[-1] for k, v in fac.items()}
    assert last["momentum"].idxmax() == "UP"                  # 动量：过去 N 期收益最高
    assert last["low_vol"].idxmax() == "QUIET"                # 低波动：负的已实现波动最大
    assert last["reversal"].idxmax() == "DROP"                # 短期反转：近期暴跌者分最高
    assert last["trend_quality"].idxmax() == "UP"             # 趋势质量：净位移/路径长度最高
    assert last["trend_quality"]["UP"] > 0.9                  # 平滑单边上涨 -> 效率比≈1
    assert last["trend_quality"]["DOWN"] < -0.9               # 平滑单边下跌 -> 效率比≈−1
    assert last["trend_quality"].abs().max() <= 1.0 + EPS
    assert last["low_vol"]["WILD"] < last["low_vol"]["QUIET"]
    # 因子只用 ≤ 当期价格：第 m 期之后全换掉，≤ m 期逐位不变
    m = 150
    fac2 = base_factor_panels(_tamper_tail(data, m), params)
    for k in FACTOR_ORDER:
        assert np.array_equal(fac[k].values[:m], fac2[k].values[:m], equal_nan=True), k
        assert not np.array_equal(fac[k].values[m], fac2[k].values[m], equal_nan=True), k


def test_cross_sectional_z_properties(synth):
    """截面 z：逐行均值 ≈ 0、标准差 ≈ 1（ddof=1）；退化行整行 NaN；与时间 shift 可交换。"""
    fac = base_factor_panels(synth, EqualWeightCompositeStrategy().params)
    for k in FACTOR_ORDER:
        z = cross_sectional_z(fac[k])
        ok = z.notna().all(axis=1)
        assert np.allclose(z[ok].mean(axis=1).values, 0.0, atol=1e-10), k
        assert np.allclose(z[ok].std(axis=1).values, 1.0, atol=1e-10), k
        # 与 shift 可交换 -> 「因子滞后一期再标准化」= 「标准化后滞后一期」，两者都只含历史
        pd.testing.assert_frame_equal(cross_sectional_z(fac[k].shift(1)), z.shift(1),
                                      atol=1e-12, rtol=0.0)
    flat = pd.DataFrame(1.0, index=synth.dates, columns=synth.symbols)   # 整行常数 -> 退化
    assert cross_sectional_z(flat).isna().all().all()
    stack = xs_normalize_stack([fac[k] for k in FACTOR_ORDER], "z")
    assert stack.shape == (len(synth.dates), len(synth.symbols), N_FACTORS)
    rank_stack = xs_normalize_stack([fac[k] for k in FACTOR_ORDER], "rank")
    assert rank_stack.shape == stack.shape
    fin = np.isfinite(rank_stack)
    assert rank_stack[fin].max() <= 0.5 + EPS and rank_stack[fin].min() >= -0.5 - EPS


def test_ic_weight_panel_proportional_to_ic_and_allows_negative():
    """IC 加权：权重 ∝ 历史 IC 均值、可为负权、Σ|w| = 1、缺历史 -> NaN、全零 -> 退回等权。"""
    ic_mean = np.array([[0.20, -0.05, 0.10, 0.00],
                        [np.nan, 0.10, 0.10, 0.10],
                        [0.00, 0.00, 0.00, 0.00]])
    ic_std = np.full((3, N_FACTORS), 0.30)
    w = ic_weight_panel(ic_mean, ic_std, scheme="ic")
    assert np.allclose(w[0], ic_mean[0] / np.abs(ic_mean[0]).sum())
    assert w[0, 0] > w[0, 2] > 0.0 > w[0, 1]                  # IC 越高权重越大，负 IC -> 负权
    assert np.isclose(np.abs(w[0]).sum(), 1.0)
    assert np.isnan(w[1]).all()                               # 任一因子缺历史 -> 整行不可用
    assert np.allclose(w[2], 1.0 / N_FACTORS)                 # IC 全为 0 -> 退回等权


def test_ic_weight_panel_ir_penalizes_unstable_ic():
    """IR 加权：IC 略低但稳定的因子权重 > IC 更高但抖动大的因子；std_floor 防爆炸。"""
    ic_mean = np.array([[0.08, 0.05]])
    ic_std = np.array([[0.40, 0.05]])
    w_ir = ic_weight_panel(ic_mean, ic_std, scheme="ir", std_floor=0.0)
    w_ic = ic_weight_panel(ic_mean, ic_std, scheme="ic")
    assert w_ir[0, 1] > w_ir[0, 0]                            # IR: 1.0 vs 0.2 -> 稳定因子胜出
    assert w_ic[0, 0] > w_ic[0, 1]                            # 纯 IC: 0.08 > 0.05
    assert np.isclose(np.abs(w_ir[0]).sum(), 1.0)
    assert np.allclose(w_ir[0], (ic_mean[0] / ic_std[0]) / np.abs(ic_mean[0] / ic_std[0]).sum())
    floor = ic_weight_panel(ic_mean, np.array([[1e-12, 0.40]]), scheme="ir", std_floor=0.02)
    assert floor[0, 0] < 1.0 - EPS and np.isclose(np.abs(floor[0]).sum(), 1.0)
    assert np.allclose(floor[0], np.array([0.08 / 0.02, 0.05 / 0.40])
                       / (0.08 / 0.02 + 0.05 / 0.40))


def test_lagged_ic_stats_shifts_before_rolling():
    """滚动统计必须在 shift 之后进行：lag 期内的 IC 不允许进入当期权重。"""
    T, F = 30, 2
    ic = np.zeros((T, F))
    ic[:, 0] = np.arange(T, dtype="float64")                  # 可辨识的递增 IC 序列
    ic[:5] = np.nan                                           # 预热
    mean, std = lagged_ic_stats(ic, lag=2, window=5, min_obs=3)
    assert np.isnan(mean[:7]).all()                           # 前 5 期 NaN + 滞后 2 期 + min_obs 3
    assert np.isfinite(mean[9, 0])
    # 第 t 行的均值 = IC[t-2-4 .. t-2] 的均值（不含 t-1、t 期）
    assert np.isclose(mean[9, 0], np.nanmean(ic[3:8, 0]))
    assert np.isclose(mean[12, 0], np.nanmean(ic[6:11, 0]))
    assert np.isclose(std[12, 0], np.nanstd(ic[6:11, 0], ddof=1))
    assert mean.shape == std.shape == (T, F)


def test_pca_weight_panel_is_first_eigenvector_of_average_correlation():
    """PCA 载荷 = 历史平均因子相关阵的第一主成分（独立复算 eigh 逐位核对）。"""
    rng = np.random.default_rng(11)
    T, N, F = 80, 7, N_FACTORS
    z = rng.standard_normal((T, N, F))
    z[:, :, 3] = -(z[:, :, 0] + 0.4 * z[:, :, 1]) + 0.2 * z[:, :, 3]   # 制造相关结构
    win, mp, lag = 20, 10, 1
    got = pca_weight_panel(z, win, mp, lag)
    assert got.shape == (T, F)
    assert np.isnan(got[:lag + mp - 1]).all()                 # 历史不足 -> NaN（下游空仓）
    for t in (lag + mp - 1, T // 2, T - 1):
        rows = range(max(0, t - win), t)                       # 滞后一期 + 尾部窗口（含不足窗口）
        A = np.stack([z[s].T @ z[s] / (N - 1) for s in rows])
        M = A.mean(axis=0)
        evals, evecs = np.linalg.eigh(0.5 * (M + M.T))
        v = evecs[:, -1]
        if v.sum() < 0:
            v = -v
        v = v / np.linalg.norm(v)
        assert np.allclose(got[t], v, atol=1e-9), t
        assert np.isclose(np.linalg.norm(got[t]), 1.0)
        assert got[t].sum() >= -1e-12                         # 符号定向：Σv ≥ 0
    # 退化输入：全 NaN / 单资产 / 零方差
    assert np.isnan(pca_weight_panel(np.full((30, 3, F), np.nan), win, mp, lag)).all()
    assert np.isnan(pca_weight_panel(z[:, :1, :], win, mp, lag)).all()
    # 另一种定向规则（按最大载荷为正）同样给出单位向量，且与默认规则至多差一个符号
    alt = pca_weight_panel(z, win, mp, lag, sign_rule="max_loading")
    rows = alt[lag + mp - 1:]
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-10)
    lead = rows[np.arange(rows.shape[0]), np.argmax(np.abs(rows), axis=1)]
    assert (lead >= 0.0).all()                                 # 最大载荷定向为正
    for t in (T // 2, T - 1):
        assert np.allclose(alt[t], got[t], atol=1e-10) or np.allclose(alt[t], -got[t], atol=1e-10)


def test_pca_loadings_on_data_are_unit_norm_and_sign_consistent(synth):
    s = PcaFactorStrategy()
    w = s.factor_weights(synth)
    t0 = s.first_active_index()
    act = w.iloc[t0:].values
    assert np.isfinite(act).all()
    assert np.allclose(np.linalg.norm(act, axis=1), 1.0, atol=1e-10)
    assert (act.sum(axis=1) >= -1e-12).all()                  # Σv ≥ 0 定向规则
    assert (np.abs(act).max(axis=1) > 1.0 / np.sqrt(N_FACTORS) - 1e-9).all()


def test_top_basket_weights_rank_linear_tilt_and_tie_break():
    """篮子配权：秩线性倾斜（k, k−1, …, 1）归一、并列按列索引稳定打破、NaN 行空仓。"""
    idx = _bdays(4)
    scores = pd.DataFrame(
        [[3.0, 1.0, 2.0, 0.5],            # n_valid=4, k=ceil(4/3)=2 -> A,C 得 2/3,1/3
         [1.0, 1.0, 0.0, -1.0],           # 并列最高 -> 按列索引取 A,B
         [np.nan] * 4,                    # 预热 -> 整行 0
         [0.1, 0.2, 0.3, 0.4]],           # 升序 -> D,C
        index=idx, columns=list("ABCD"))
    w = top_basket_weights(scores, 1.0 / 3.0)
    assert np.allclose(w.iloc[0].values, [2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0])
    assert np.allclose(w.iloc[1].values, [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
    assert (w.iloc[2].values == 0.0).all()
    assert np.allclose(w.iloc[3].values, [0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(w.sum(axis=1).values, [1.0, 1.0, 0.0, 1.0], atol=SUM_EPS)
    assert (w.values >= 0.0).all() and w.values.max() <= 1.0 + EPS
    # k=3 时权重为 3/6, 2/6, 1/6（秩线性倾斜，只用序信息）
    w2 = top_basket_weights(scores.iloc[[0]], 0.75)
    assert np.allclose(np.sort(w2.values[0])[::-1][:3], [0.5, 1.0 / 3.0, 1.0 / 6.0])


# ------------------------------------------------------------------ 行为断言

def test_equalweight_composite_prefers_multi_factor_dominant_asset():
    """强动量 + 低波动 + 高趋势质量三维占优的 STAR，必须拿到最高权重（篮子第一名）。"""
    data = _star_universe()
    s = EqualWeightCompositeStrategy()
    fac = s.factor_panels(data)
    score = s.composite_scores(data)
    w = s.generate_weights(data)
    t0 = s.first_active_index()
    last_z = {k: cross_sectional_z(v).iloc[-1] for k, v in fac.items()}
    # STAR 在动量/低波/趋势质量三维都排第一（短期反转维度不占优也不影响合成结论）
    for k in ("momentum", "low_vol", "trend_quality"):
        assert last_z[k].idxmax() == "STAR", k                   # 三维都是截面第一
        assert last_z[k]["STAR"] > 0.5, (k, last_z[k]["STAR"])   # 且明显高于截面均值
        assert last_z[k]["STAR"] == last_z[k].max(), k
    assert score.iloc[-1].idxmax() == "STAR"
    assert w.iloc[-1]["STAR"] == w.iloc[-1].max()
    assert np.isclose(w.iloc[-1]["STAR"], 2.0 / 3.0, atol=EPS)     # 6 资产 k=2 -> 第一名 2/3
    assert w.iloc[-1]["CHOPPY"] == 0.0                             # 震荡高波资产不入篮子
    act = w.iloc[t0:]
    assert (act.idxmax(axis=1) == "STAR").all()                    # 每个交易日都是第一名
    assert (act["STAR"] > 0.0).all()
    # 等权复合权重恒为 1/F（不依赖任何历史标签）
    assert np.allclose(s.factor_weights(data).values, 1.0 / N_FACTORS)


def test_pca_composite_also_ranks_dominant_asset_top():
    """PCA 复合（载荷按 Σv ≥ 0 定向）同样把多因子占优的 STAR 排在篮子第一。"""
    data = _star_universe()
    s = PcaFactorStrategy()
    w = s.generate_weights(data)
    act = w.iloc[s.first_active_index():]
    assert (act.idxmax(axis=1) == "STAR").all()
    assert w.iloc[-1]["STAR"] == w.iloc[-1].max() > 0.0
    load = s.factor_weights(data).iloc[-1]
    assert load.sum() >= 0.0                                       # 定向规则生效
    assert load["momentum"] > 0.0 and load["trend_quality"] > 0.0  # 共识方向为正载荷


def test_ic_weighted_prefers_high_ic_factor():
    """趋势市里：历史 IC 最高的动量因子拿最大权重，IC 为负的短期反转因子拿负权。"""
    data = _drift_universe()
    s = IcWeightedCompositeStrategy()
    w = s.factor_weights(data)
    ic_mean, ic_std = s.ic_stats(data)
    t0 = s.first_active_index()
    act_w = w.iloc[t0:]
    act_ic = ic_mean.iloc[t0:]
    assert np.isfinite(act_w.values).all()
    assert np.allclose(np.abs(act_w.values).sum(axis=1), 1.0, atol=1e-9)
    last_w, last_ic = w.iloc[-1], ic_mean.iloc[-1]
    assert last_ic.idxmax() == "momentum"                          # 构造数据里动量 IC 最高
    assert last_w.idxmax() == "momentum" and last_w["momentum"] > 0.0
    assert last_w.idxmin() == "reversal" and last_w["reversal"] < 0.0   # 负 IC -> 负权（反向使用）
    assert last_ic["reversal"] < 0.0
    # 逐行核对：权重排序与 IC 均值排序完全一致（IC 越高权重越大）
    assert (np.argsort(-act_w.values, axis=1, kind="stable")
            == np.argsort(-act_ic.values, axis=1, kind="stable")).all()
    # 权重与 IC 均值成正比（Σ|w| = 1 归一后）
    ref = act_ic.values / np.abs(act_ic.values).sum(axis=1, keepdims=True)
    assert np.allclose(act_w.values, ref, atol=1e-12)


def test_max_ir_prefers_stable_factor():
    """IR 加权：逐行权重排序与 mean(IC)/std(IC) 排序一致，且稳定性会改写 IC 的排序。"""
    data = _drift_universe()
    s = MaxIrCompositeStrategy()
    w = s.factor_weights(data)
    ic_mean, ic_std = s.ic_stats(data)
    t0 = s.first_active_index()
    act_w = w.iloc[t0:]
    ir = (ic_mean / ic_std.clip(lower=float(s.params["ir_std_floor"]))).iloc[t0:]
    assert np.isfinite(act_w.values).all()
    assert np.allclose(np.abs(act_w.values).sum(axis=1), 1.0, atol=1e-9)
    assert (np.argsort(-act_w.values, axis=1, kind="stable")
            == np.argsort(-ir.values, axis=1, kind="stable")).all()
    ref = ir.values / np.abs(ir.values).sum(axis=1, keepdims=True)
    assert np.allclose(act_w.values, ref, atol=1e-12)
    # IR 与纯 IC 加权必须给出**不同**的权重（分母的方差惩罚确实生效）
    ic_w = IcWeightedCompositeStrategy().factor_weights(data).iloc[t0:]
    assert not np.allclose(act_w.values, ic_w.values, atol=1e-6)
    assert np.abs(act_w.values - ic_w.values).max() > 1e-4


def test_ic_and_ir_composites_stay_long_only_and_invested(synth):
    """IC/IR 权重可为负，但复合分排序后的**组合权重**仍严格只做多且满仓。"""
    for cls in IC_STRATEGIES:
        s = cls()
        fw = s.factor_weights(synth)
        w = s.generate_weights(synth)
        assert np.nanmin(fw.values) < 0.0, f"{cls.name} 从未出现负因子权重"
        assert (w.values >= -EPS).all(), cls.name
        act = w.iloc[s.first_active_index():]
        assert np.allclose(act.sum(axis=1).values, 1.0, atol=SUM_EPS), cls.name


def test_rank_normalization_mode_also_works():
    """截面标准化支持 rank 模式（百分位排名去中值），复合与配权逻辑不变。"""
    data = _star_universe()
    s = EqualWeightCompositeStrategy()
    s.params = {**s.params, "xs_norm": "rank"}                  # 实例级覆盖，不改类属性
    w = s.generate_weights(data)
    t0 = s.first_active_index()
    act = w.iloc[t0:]
    assert np.allclose(act.sum(axis=1).values, 1.0, atol=SUM_EPS)
    assert (w.values >= -EPS).all() and np.isfinite(w.values).all()
    assert (act.idxmax(axis=1) == "STAR").mean() > 0.9           # 秩较粗，STAR 绝大多数日第一
    assert w.iloc[-1].idxmax() == "STAR"


# ------------------------------------------------------------------ 边界与集成

def test_degenerate_universes_are_safe():
    """单资产 / 超短历史 / 两资产 / 全常数价格 / 重复列 都必须安全返回合法权重。"""
    base = make_synthetic_universe(n_assets=4, n_days=400, seed=5)
    idx = base.prices.index
    cases = {
        "one": MarketData(prices=base.prices[["A0"]]),
        "short": MarketData(prices=base.prices.iloc[:10]),
        "two": MarketData(prices=base.prices[["A0", "A1"]]),
        "flat": MarketData(prices=pd.DataFrame(100.0, index=idx[:200], columns=list("ABCD"))),
        "dup": MarketData(prices=pd.DataFrame(
            {"A": base.prices["A0"].values, "B": base.prices["A0"].values,
             "C": base.prices["A1"].values}, index=idx)),
    }
    for tag, d in cases.items():
        for cls in STRATEGIES:
            s = cls()
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, (tag, cls.name)
            assert list(w.columns) == d.symbols and w.index.equals(d.dates), (tag, cls.name)
            assert np.isfinite(w.values).all(), (tag, cls.name)
            assert w.values.min() >= -EPS and w.values.max() <= 1.0 + EPS, (tag, cls.name)
            row_sum = w.sum(axis=1).values
            assert np.all(np.isclose(row_sum, 0.0, atol=1e-12)
                          | np.isclose(row_sum, 1.0, atol=SUM_EPS)), (tag, cls.name)
            sc = s.composite_scores(d).values                   # 复合分不得出现 inf
            assert not np.isinf(sc).any(), (tag, cls.name)
            fw = s.factor_weights(d).values                     # 因子权重面板同样安全
            assert not np.isinf(fw).any(), (tag, cls.name)
    # 单资产（截面标准差无定义）与全常数价格（无截面离散度）必须整段空仓
    for tag in ("one", "flat"):
        for cls in STRATEGIES:
            w = cls().generate_weights(cases[tag])
            assert (w.values == 0.0).all(), (tag, cls.name)


def test_only_numpy_pandas_dependencies():
    """依赖白名单：本渠道只用 numpy/pandas（因子/IC/PCA/配权工具全部自实现）。"""
    import kairos_strategies.channels.multi_factor as mod
    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "pandas", "typing", "__future__"}, imported
    for bad in ("statsmodels", "sklearn", "scipy"):
        assert bad not in mod.__dict__, bad


def test_discover_registers_channel_with_unique_names():
    """全局注册：4 个策略被 discover 找到、channel 分组正确、名字全局唯一不重复。"""
    all_strats = discover()
    names = [s.name for s in all_strats]
    assert len(names) == len(set(names)), "存在重名策略"
    assert EXPECTED_NAMES <= set(names)
    grouped = [s for s in all_strats if s.channel == "multi_factor"]
    assert {s.name for s in grouped} == EXPECTED_NAMES


def test_engine_accepts_composite_weights(synth):
    """复合权重可直接进引擎：收益/换手有限、换手不虚增（≤ 2）。"""
    bt = Backtester()
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        res = bt.run(synth, w)
        assert np.isfinite(res.returns.values).all(), cls.name
        assert np.isfinite(res.turnover.values).all(), cls.name
        assert res.turnover.max() <= 2.0 + 1e-6, (cls.name, res.turnover.max())
        assert len(res.metrics) > 0, cls.name


def test_composite_scores_finite_and_aligned(synth):
    """复合分面板与 data 对齐；预热期 NaN、预热后有限（NaN 会被下游当成不选）。"""
    for cls in STRATEGIES:
        s = cls()
        sc = s.composite_scores(synth)
        assert sc.shape == (len(synth.dates), len(synth.symbols)), cls.name
        assert list(sc.columns) == synth.symbols
        t0 = s.first_active_index()
        assert sc.iloc[:t0].isna().all().all(), cls.name
        assert np.isfinite(sc.iloc[t0:].values).all(), cls.name
