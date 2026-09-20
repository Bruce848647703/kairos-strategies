"""pairs 渠道测试。

覆盖：形状/列对齐/有限值、**每行权重和 ≈ 0（美元中性）**、每行绝对值和 ≤ 1+eps、
值域 [-1,1]、确定性（同数据/同参数不同实例）、预热期空仓、退化 universe 边界、
依赖白名单（只用 numpy/pandas，且不得 import 兄弟渠道的私有函数）、
**防未来函数**（篡改 t 之后的价格，t 及之前的权重逐位不变；并附带截断样本的前缀不变性），
以及若干**行为断言**：

  * ssd_pairs：构造「两只高度协整（价差平稳）+ 第三只独立随机游走」，验证 SSD 距离法
    选中协整对（随机游走腿权重恒为 0）；价差走阔 -> 做空贵腿/做多便宜腿（符号相反），
    价差收窄 -> 反向。
  * coint_pairs_portfolio：构造 4 组互不相关的协整对（8 只资产），验证**多对分散**
    （同一行出现 ≥ 2 对、预算等分、每对两腿等幅反向）、只挑协整对（随机游走腿不交易）、
    方向跟随价差（走阔的一对被做空贵腿、收窄的一对被做多便宜腿）；并用「8 只独立随机
    游走」的对照 universe 验证协整门槛真的在筛（活跃度显著更低）。
  * sector_neutral_pairs：构造 3 个板块（各 2 只、由独立板块因子驱动），验证层次聚类
    **还原出构造的板块**、配对只发生在板块内部、**每个板块的净敞口 ≈ 0**、整体行和 ≈ 0。
"""
from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, make_synthetic_universe
from kairos_strategies.channels import pairs as mod
from kairos_strategies.channels.pairs import (
    CointPairsPortfolioStrategy,
    SectorNeutralPairsStrategy,
    SsdPairsStrategy,
    _adf_diagnostics,
    _agglomerative_groups,
    _corr_matrix,
    _dollar_neutral,
    _rebalance_grid,
    _signal_ramp,
    _ssd_matrix,
    _variance_ratio,
)

STRATEGIES = [SsdPairsStrategy, CointPairsPortfolioStrategy, SectorNeutralPairsStrategy]
EXPECTED_NAMES = {"ssd_pairs", "coint_pairs_portfolio", "sector_neutral_pairs"}
EPS = 1e-9            # 绝对值和 / 值域容差
SUM_EPS = 1e-9        # 每行权重和（美元中性）容差
NEUTRAL_EPS = 1e-9    # 板块净敞口容差
CJK = re.compile(r"[\u4e00-\u9fff]")


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """平稳 AR(1)/离散 OU 残差（协整对的「价差」原型）。"""
    e = np.zeros(n, dtype="float64")
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.standard_normal() * sigma
    return e


def _shock(n: int, tail: int, size: float) -> np.ndarray:
    """末段 ``tail`` 天的水平冲击（模拟价差突然走阔/收窄）。"""
    s = np.zeros(n, dtype="float64")
    if tail > 0 and size:
        s[-tail:] = size
    return s


def _coint_universe(shock: float = 0.20, n: int = 266, tail: int = 3,
                    third: bool = True, seed: int = 7) -> MarketData:
    """A0/A1 高度协整（logy = 0.3 + 1.1·logx + 平稳 AR(1) 残差），A2 独立随机游走。

    ``shock > 0`` 让 A1 在末段相对 A0 **走阔**（A1 变贵），``shock < 0`` 则收窄。
    n = 266 是刻意选的：末段冲击落在**最后一次再平衡的估计窗口之外**（选对/估 β 用的是
    干净样本），而价差 z-score 能看到冲击 —— 这正是实盘时序，也避免因冲击污染窗口而
    让协整门槛拒绝对（那不是本测试要检验的东西）。
    """
    rng = np.random.default_rng(seed)
    logx = np.log(100.0) + np.cumsum(rng.standard_normal(n) * 0.010)
    logy = 0.3 + 1.1 * logx + _ar1(n, 0.85, 0.008, rng)
    cols = {"A0": np.exp(logx),
            "A1": np.exp(logy + _shock(n, tail, shock))}
    if third:
        logz = np.log(80.0) + np.cumsum(rng.standard_normal(n) * 0.015)   # 独立随机游走
        cols["A2"] = np.exp(logz)
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _multi_pair_universe(n: int = 414, shock: float = 0.20, tail: int = 3,
                         seed: int = 21) -> MarketData:
    """4 组互不相关的协整对（8 只资产）：(A0,A1) (A2,A3) (A4,A5) (A6,A7)。

    每对由一个独立的共同因子驱动、β 各不相同（1.05/1.12/1.19/1.26），残差是平稳 AR(1)。
    末段让 **A1 走阔**（+shock）、**A5 收窄**（-shock），用于检验组合的方向与分散化。
    n = 414 同样保证冲击落在最后一个 block 的估计窗口之外。
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for k in range(4):
        factor = np.cumsum(rng.standard_normal(n) * 0.012)
        logx = np.log(60.0 + 10.0 * k) + factor
        logy = 0.2 + 0.05 * k + (1.05 + 0.07 * k) * logx + _ar1(n, 0.86, 0.008, rng)
        cols[f"A{2 * k}"] = np.exp(logx)
        cols[f"A{2 * k + 1}"] = np.exp(logy)
    cols["A1"] = cols["A1"] * np.exp(_shock(n, tail, +shock))
    cols["A5"] = cols["A5"] * np.exp(_shock(n, tail, -shock))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _random_walk_universe(n: int = 420, n_assets: int = 8, seed: int = 99) -> MarketData:
    """对照组：``n_assets`` 只**互相独立**的随机游走（本就不该有协整对）。"""
    rng = np.random.default_rng(seed)
    logp = np.log(100.0) + np.cumsum(rng.standard_normal((n, n_assets)) * 0.015, axis=0)
    cols = [f"A{i}" for i in range(n_assets)]
    return MarketData(prices=pd.DataFrame(np.exp(logp), index=_bdays(n), columns=cols))


def _sector_universe(n: int = 360, shock_col: str = "A3", shock: float = 0.15,
                     tail: int = 3, seed: int = 31) -> MarketData:
    """3 个板块 × 2 只：{A0,A1} {A2,A3} {A4,A5}，各板块由独立因子驱动。

    板块内两只共享同一因子（相关系数 ≈ 1）、跨板块因子独立（相关系数 ≈ 0），
    因此「正确的板块划分」是唯一确定的。末段让 ``shock_col`` 相对同板块的另一只走阔。
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for s in range(3):
        factor = np.cumsum(rng.standard_normal(n) * 0.015)
        for m in range(2):
            i = 2 * s + m
            beta = 1.0 if m == 0 else 1.05
            cols[f"A{i}"] = np.exp(np.log(70.0 + 8.0 * i) + beta * factor
                                   + _ar1(n, 0.90, 0.004, rng))
    cols[shock_col] = cols[shock_col] * np.exp(_shock(n, tail, shock))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2019-01-01")))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=8, n_days=600, seed=2026)


# ------------------------------------------------------------------ 内部断言工具

def _assert_neutral(w: pd.DataFrame, tag: str) -> None:
    """美元中性硬约束：行和 ≈ 0、毛敞口 ≤ 1、值域 [-1,1]、无 NaN。"""
    assert np.isfinite(w.values).all(), tag
    assert np.abs(w.sum(axis=1)).max() <= SUM_EPS, (tag, float(np.abs(w.sum(axis=1)).max()))
    assert w.abs().sum(axis=1).max() <= 1.0 + EPS, (tag, float(w.abs().sum(axis=1).max()))
    assert w.values.min() >= -1.0 - EPS and w.values.max() <= 1.0 + EPS, tag


def _assert_antisymmetric(w: pd.DataFrame, tag: str) -> None:
    """每一行的非零权重都能配成「等幅反向」的两腿（配对组合的结构不变量）。"""
    v = w.to_numpy(dtype="float64")
    for row in v:
        nz = np.sort(row[row != 0.0])
        assert nz.size % 2 == 0, (tag, nz)
        assert np.allclose(nz, -nz[::-1], atol=EPS, rtol=0.0), (tag, nz)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "pairs"
        assert s.universe == "cross_section"
        assert s.long_only is False                          # 配对交易天然多空
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
            assert CJK.search(meta[key]), (s.name, key)      # 元信息为中文
        assert isinstance(meta["params"], dict) and meta["params"]
        assert all(isinstance(k, str) for k in meta["params"]), s.name
    assert names == EXPECTED_NAMES


def test_shape_columns_index_alignment(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == (len(synth.dates), len(synth.symbols)), cls.name
        assert list(w.columns) == synth.symbols
        assert w.index.equals(synth.dates)


def test_all_values_finite_and_neutral(synth):
    for cls in STRATEGIES:
        _assert_neutral(cls().generate_weights(synth), cls.name)


def test_row_sum_is_zero_on_every_universe():
    """行和 ≈ 0（±1e-9）在本渠道用到的所有 universe 上都必须成立（含空仓行）。"""
    universes = [make_synthetic_universe(n_assets=8, n_days=400, seed=3),
                 _coint_universe(shock=0.20), _multi_pair_universe(),
                 _random_walk_universe(n=300), _sector_universe()]
    for cls in STRATEGIES:
        s = cls()
        for d in universes:
            w = s.generate_weights(d)
            _assert_neutral(w, f"{cls.name}/{d.prices.shape}")
            _assert_antisymmetric(w, f"{cls.name}/{d.prices.shape}")


def test_long_short_both_signs_appear(synth):
    """可多空策略必须真的同时产生多头与空头，且每个非空仓行都两腿俱全。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() < 0.0, f"{cls.name} 从未做空"
        assert w.values.max() > 0.0, f"{cls.name} 从未做多"
        rows = w[w.abs().sum(axis=1) > 0]
        assert len(rows) >= 50, (cls.name, len(rows))
        assert (rows.max(axis=1) > 0).all() and (rows.min(axis=1) < 0).all(), cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)


def test_warmup_rows_are_flat(synth):
    """信息不足的预热期必须整行空仓，且预热长度与窗口参数一致。"""
    p = CointPairsPortfolioStrategy.params
    cases = {
        SsdPairsStrategy: SsdPairsStrategy.params["formation_window"] - 1,
        CointPairsPortfolioStrategy: max(p["est_window"], mod._MIN_OBS + p["adf_lags"] + 2) - 1,
        SectorNeutralPairsStrategy: max(SectorNeutralPairsStrategy.params["corr_window"],
                                        SectorNeutralPairsStrategy.params["est_window"]),
    }
    for cls, warm in cases.items():
        w = cls().generate_weights(synth)
        assert (w.iloc[:warm].values == 0.0).all(), (cls.name, warm)


def test_no_lookahead_tamper_invariance(synth):
    """**防未来函数**：篡改 t 之后的价格，t 及之前的权重必须逐位（bitwise）不变。

    再平衡网格锚定在样本起点、所有估计只用尾部窗口，故未来数据的任何改动都不会
    改写历史权重。同时校验「篡改确实生效」（否则该断言是空转）。
    """
    rng = np.random.default_rng(123)
    t = 300
    tampered = synth.prices.copy()
    block = tampered.iloc[t + 1:]
    tampered.iloc[t + 1:] = block * rng.uniform(0.6, 1.7, size=block.shape) + 5.0
    assert not np.allclose(tampered.values, synth.prices.values)
    d2 = MarketData(prices=tampered, volumes=synth.volumes,
                    periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth).values
        w2 = s.generate_weights(d2).values
        assert np.array_equal(w1[:t + 1], w2[:t + 1]), cls.name          # 逐位不变
        assert not np.array_equal(w1, w2), f"{cls.name} 未来数据未生效？"
        assert np.abs(w1[t + 1:] - w2[t + 1:]).max() > 1e-6, cls.name
        _assert_neutral(pd.DataFrame(w2, index=synth.dates, columns=synth.symbols), cls.name)


def test_no_lookahead_labels_and_prefix_invariance(synth):
    """截断样本重算：前缀权重与板块标签都不变（网格锚定起点的另一面）。"""
    m = 420
    sub = MarketData(prices=synth.prices.iloc[:m], volumes=None,
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.allclose(full, part, atol=1e-12, rtol=0.0), cls.name
        assert np.allclose(full[:m - 1], part[:m - 1], atol=0.0, rtol=0.0), cls.name
    lab_full = SectorNeutralPairsStrategy().sector_labels(synth, at=m - 1)
    lab_part = SectorNeutralPairsStrategy().sector_labels(sub)
    assert np.array_equal(lab_full, lab_part)


def test_only_numpy_pandas_and_self_implemented_helpers():
    """依赖白名单 + 不得 import 兄弟渠道的私有函数（辅助函数必须在本模块自实现）。"""
    tree = ast.parse(inspect.getsource(mod))
    top, rel = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    top.add(node.module.split(".")[0])
            else:
                rel.add(node.module or "")                 # from .. import indicators
                rel.update(a.name for a in node.names)     # from ..base import MarketData
    assert top <= {"numpy", "pandas", "typing", "__future__"}, top
    assert rel <= {"", "base", "indicators", "MarketData", "Strategy"}, rel
    for bad in ("statsmodels", "sklearn", "scipy", "statarb", "meanrev"):
        assert bad not in mod.__dict__, bad
        assert not any(bad in r for r in rel), (bad, rel)
    for fn in ("_dollar_neutral", "_signal_ramp", "_ssd_matrix", "_corr_matrix",
               "_adf_diagnostics", "_variance_ratio", "_agglomerative_groups",
               "_rebalance_grid", "_ols_ab", "_ols_t_stats"):
        assert getattr(mod, fn).__module__ == mod.__name__, fn   # 本模块自实现


def test_degenerate_universes_are_safe():
    """单资产 / 历史过短 / 两资产 / 常数价格 等退化输入都必须安全返回。"""
    base = make_synthetic_universe(n_assets=3, n_days=300, seed=5)
    flat = MarketData(prices=pd.DataFrame(100.0, index=_bdays(200), columns=["A0", "A1", "A2"]))
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    for cls in STRATEGIES:
        s = cls()
        for d in (one, short, flat):
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, cls.name
            assert (w.values == 0.0).all(), (cls.name, d.prices.shape)   # 凑不出多空 -> 空仓
        w2 = s.generate_weights(two)
        assert w2.shape == two.prices.shape
        _assert_neutral(w2, cls.name)


def test_engine_accepts_pairs_weights(synth):
    """行和为 0 的配对权重可直接进引擎：换手有限且不虚增。"""
    bt = Backtester()
    for cls in STRATEGIES:
        res = bt.run(synth, cls().generate_weights(synth))
        assert np.isfinite(res.returns.values).all(), cls.name
        assert np.isfinite(res.turnover.values).all(), cls.name
        assert res.turnover.max() <= 2.0 + 1e-6, (cls.name, float(res.turnover.max()))


# ------------------------------------------------------------------ 行为断言

def test_ssd_pairs_picks_similar_legs_and_trades_against_widening():
    """SSD 距离法必须选中「走势最像」的协整对，并在价差走阔/收窄时反向持仓。"""
    s = SsdPairsStrategy()
    wide = s.generate_weights(_coint_universe(shock=+0.20))
    narrow = s.generate_weights(_coint_universe(shock=-0.20))
    for w in (wide, narrow):
        assert (w["A2"].values == 0.0).all()                 # 随机游走腿从未被选中
        assert (w[["A0", "A1"]].values != 0.0).any()          # 协整对被选中并交易
        per_row = (w.values != 0.0).sum(axis=1)
        assert set(np.unique(per_row)) <= {0, 2}, np.unique(per_row)   # 任意时刻只有两腿
    last_w, last_n = wide.iloc[-1], narrow.iloc[-1]
    assert last_w["A1"] < 0.0 < last_w["A0"]                  # A1 变贵 -> 做空 A1 / 做多 A0
    assert last_n["A1"] > 0.0 > last_n["A0"]                  # A1 变便宜 -> 反向
    for last in (last_w, last_n):
        assert np.isclose(last["A0"], -last["A1"], atol=EPS)  # 两腿等预算反向
        assert abs(last["A0"]) + abs(last["A1"]) <= 1.0 + EPS
    _assert_neutral(wide, "ssd/wide")
    _assert_neutral(narrow, "ssd/narrow")


def test_ssd_closest_pair_helper_is_ssd_minimiser():
    """``_closest_pair`` 必须返回 SSD 最小的那一对，且退化列被排除。"""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((60, 4))
    win = np.column_stack([base[:, 0], base[:, 0] + 1e-4 * base[:, 1],
                           rng.standard_normal(60) * 5.0, np.ones(60)])
    pair = SsdPairsStrategy._closest_pair(win)
    assert pair == (0, 1), pair                              # 第 0/1 列几乎重合
    d = _ssd_matrix(win)
    off_diag = d[~np.eye(win.shape[1], dtype=bool)]
    assert np.isclose(d[pair], float(off_diag.min()), atol=0.0)
    assert SsdPairsStrategy._closest_pair(np.ones((60, 3))) is None      # 全常数 -> None
    assert SsdPairsStrategy._closest_pair(np.ones((60, 1))) is None      # 单资产 -> None


def test_ssd_matrix_is_symmetric_and_exact():
    A = np.array([[1.0, 2.0, 4.0], [1.5, 1.0, 3.0], [2.0, 0.0, 2.0]])
    d = _ssd_matrix(A)
    assert np.allclose(d, d.T, atol=0.0)
    assert np.allclose(np.diag(d), 0.0, atol=0.0)
    assert (d >= 0.0).all()
    for i in range(3):                                       # 与逐对定义完全一致
        for j in range(3):
            assert np.isclose(d[i, j], float(((A[:, i] - A[:, j]) ** 2).sum()), atol=1e-12)


def test_coint_portfolio_spreads_budget_over_multiple_pairs():
    """协整配对**组合**：同一 block 内选中多对、腿不重复、预算等分到每一对。"""
    s = CointPairsPortfolioStrategy()
    d = _multi_pair_universe()
    w = s.generate_weights(d)
    p = s.params
    est = max(int(p["est_window"]), mod._MIN_OBS + int(p["adf_lags"]) + 2)
    grid = _rebalance_grid(len(d.dates), est - 1, int(p["refit_freq"]))
    L = mod._log_panel(d)
    counts = [len(s._select_pairs(L[t0 - est + 1:t0 + 1])) for t0 in grid]
    assert max(counts) >= 2, counts                          # 组合里真的同时持有多对
    assert sum(c >= 2 for c in counts) >= len(counts) // 2, counts
    assert max(counts) <= int(p["max_pairs"])                # 不超上限

    t0 = grid[-1]
    chosen = s._select_pairs(L[t0 - est + 1:t0 + 1])
    assert len(chosen) >= 2, chosen
    legs = [k for pr in chosen for k in pr[0]]
    assert len(set(legs)) == len(legs)                       # exclusive_legs：腿不重复
    per_leg = float(p["gross_budget"]) / (2.0 * len(chosen))  # 预算等分到每对、两腿各半
    block = w.iloc[t0:]
    for (i, j), _alpha, _beta in chosen:
        a = block.iloc[:, i].to_numpy(dtype="float64")
        b = block.iloc[:, j].to_numpy(dtype="float64")
        assert np.allclose(a, -b, atol=0.0, rtol=0.0), (i, j)  # 两腿严格等幅反向
        assert np.abs(b).max() <= per_leg + 1e-12              # 单腿不超预算
        assert np.isclose(np.abs(b).max(), per_leg, atol=1e-9)  # 价差走阔时打满强度
    others = [k for k in range(len(d.symbols))
              if k not in {x for pr in chosen for x in pr[0]}]
    assert (block.iloc[:, others].to_numpy() == 0.0).all()      # 未入选的资产不交易
    gross = float(block.abs().sum(axis=1).max())
    assert gross <= float(p["gross_budget"]) + EPS              # 毛敞口不超预算
    assert gross >= 0.95 * min(1.0, per_leg * 2 * len(chosen))  # 价差走阔时几乎打满预算
    _assert_neutral(w, "coint_portfolio")
    _assert_antisymmetric(w, "coint_portfolio")
    assert (w.abs().sum(axis=1) > 0).sum() >= 50


def test_coint_portfolio_direction_follows_spread():
    """走阔的一对（A1 变贵）-> 做空 A1/做多 A0；收窄的一对（A5 变便宜）-> 做多 A5/做空 A4。"""
    s = CointPairsPortfolioStrategy()
    last = s.generate_weights(_multi_pair_universe(shock=+0.20)).iloc[-1]
    assert last["A1"] < 0.0 < last["A0"]
    assert last["A5"] > 0.0 > last["A4"]
    flipped = s.generate_weights(_multi_pair_universe(shock=-0.20)).iloc[-1]
    assert flipped["A1"] > 0.0 > flipped["A0"]               # 冲击反向 -> 持仓反向


def test_coint_portfolio_selects_cointegrated_pair_not_random_walk():
    """「两协整腿 + 一随机游走」：组合只交易协整对，随机游走腿权重恒为 0。"""
    s = CointPairsPortfolioStrategy()
    w = s.generate_weights(_coint_universe(shock=+0.20))
    assert (w["A2"].values == 0.0).all()
    assert w.iloc[-1]["A1"] < 0.0 < w.iloc[-1]["A0"]         # 价差走阔 -> 做空贵腿
    assert np.isclose(w.iloc[-1]["A0"], -w.iloc[-1]["A1"], atol=EPS)
    _assert_neutral(w, "coint_portfolio/3assets")


def test_coint_portfolio_screening_rejects_pure_random_walks():
    """双重门槛（ADF t + 方差比）必须真的在筛：独立随机游走 universe 上活跃度显著更低。"""
    s = CointPairsPortfolioStrategy()
    w_rw = s.generate_weights(_random_walk_universe())
    w_ci = s.generate_weights(_multi_pair_universe())
    act_rw = float((w_rw.abs().sum(axis=1) > 0).mean())
    act_ci = float((w_ci.abs().sum(axis=1) > 0).mean())
    assert act_rw < 0.5 * act_ci, (act_rw, act_ci)
    for w in (w_rw, w_ci):
        _assert_neutral(w, "coint_portfolio/screening")


def test_coint_diagnostics_separate_stationary_from_random_walk():
    """自研平稳性诊断本身要有效：平稳残差的 ADF t 更负、方差比更低、半衰期更短。"""
    rng = np.random.default_rng(11)
    n = 200
    x = np.cumsum(rng.standard_normal(n) * 0.015)
    y_rw = np.cumsum(rng.standard_normal(n) * 0.015)                  # 不协整
    y_ci = 0.4 + 1.2 * x + _ar1(n, 0.85, 0.008, rng)                  # 协整
    out = {}
    for tag, y in (("rw", y_rw), ("ci", y_ci)):
        a, b = mod._ols_ab(x, y)
        e = y - a - b * x
        t, rho, hl = _adf_diagnostics(e, lags=2)
        out[tag] = (t, rho, hl, _variance_ratio(e, 5))
    assert out["ci"][0] < out["rw"][0] - 1.0                 # 协整的 t 明显更负
    assert out["ci"][3] < out["rw"][3]                        # 协整的方差比更低
    assert out["ci"][1] < 1.0 <= out["rw"][1] + 0.5           # AR(1) 系数
    assert out["ci"][2] < out["rw"][2]                        # 半衰期更短
    assert _variance_ratio(np.arange(50, dtype="float64"), 5) > 1.0    # 趋势 -> VR > 1
    assert not np.isfinite(_variance_ratio(np.zeros(50), 5))           # 退化 -> inf
    assert _adf_diagnostics(np.zeros(50), 2)[2] == np.inf              # 退化 -> 无回归


def test_sector_neutral_recovers_constructed_sectors():
    """层次聚类必须还原出构造的 3 个板块，且是全体资产的一个划分。"""
    s = SectorNeutralPairsStrategy()
    d = _sector_universe()
    labels = s.sector_labels(d)
    groups = {int(k): sorted(np.asarray(d.symbols)[labels == k].tolist())
              for k in np.unique(labels)}
    assert len(groups) == 3, groups
    got = sorted(tuple(v) for v in groups.values())
    assert got == [("A0", "A1"), ("A2", "A3"), ("A4", "A5")], got
    assert sorted(c for v in groups.values() for c in v) == list(d.symbols)   # 无遗漏/无重叠
    assert (s.sector_labels(d, at=0) == -1).all()              # 预热期还没有划分


def test_sector_neutral_net_exposure_is_zero_per_sector():
    """**板块中性**：每个板块的净敞口 ≈ 0，配对只发生在板块内部，整体行和 ≈ 0。"""
    s = SectorNeutralPairsStrategy()
    d = _sector_universe(shock_col="A3", shock=+0.15)
    w = s.generate_weights(d)
    labels = s.sector_labels(d)
    sectors = [list(np.asarray(d.symbols)[labels == k]) for k in np.unique(labels)]
    tail = w.iloc[-30:]                                        # 最后一个 block 内板块冻结
    for cols in sectors:
        net = tail[cols].sum(axis=1)
        assert np.abs(net).max() <= NEUTRAL_EPS, (cols, float(np.abs(net).max()))
        nz = (tail[cols].values != 0.0).sum(axis=1)
        assert set(np.unique(nz)) <= {0, 2}, (cols, np.unique(nz))   # 块内恰好一对两腿
    _assert_neutral(w, "sector_neutral")
    _assert_antisymmetric(w, "sector_neutral")
    last = w.iloc[-1]
    assert last["A3"] < 0.0 < last["A2"]                        # A3 走阔 -> 做空 A3/做多 A2
    assert np.isclose(last["A2"], -last["A3"], atol=EPS)
    assert np.isclose(last["A0"], -last["A1"], atol=EPS)        # 其它板块同样两腿等幅反向
    for cols in sectors:                                        # 板块间预算等分
        assert np.isclose(tail[cols].abs().sum(axis=1).max(),
                          w.abs().sum(axis=1).max() / len(sectors), atol=1e-9)


def test_sector_neutral_never_pairs_across_sectors():
    """任何一条腿的反向对手腿都在**同一板块**内（配对候选被板块结构约束）。"""
    s = SectorNeutralPairsStrategy()
    d = _sector_universe()
    w = s.generate_weights(d)
    labels = s.sector_labels(d)
    p = s.params
    start = max(int(p["corr_window"]), int(p["est_window"]))
    grid = _rebalance_grid(len(d.dates), start, int(p["refit_freq"]))
    # 本 universe 的板块划分在整个样本上稳定（否则下面按「末期标签」核对会失真）
    for t0 in grid:
        assert np.array_equal(s.sector_labels(d, at=t0), labels), t0
    v = w.to_numpy(dtype="float64")
    seen = 0
    for row in v:
        nz = np.flatnonzero(row != 0.0)
        if nz.size == 0:
            continue
        seen += 1
        assert nz.size % 2 == 0, nz
        for c in nz:
            peers = [k for k in nz if k != c and labels[k] == labels[c]]
            assert len(peers) == 1, (int(c), labels[nz])       # 同板块恰好一个对手腿
            assert np.isclose(row[c], -row[peers[0]], atol=EPS)
        for k in np.unique(labels):                            # 每块贡献 0 或 2 条腿
            assert int((labels[nz] == k).sum()) in (0, 2), (int(k), nz)
    assert seen >= 50, seen


def test_agglomerative_groups_partition_and_determinism():
    """聚类是自研的平均链接：结果为划分、可复现、并列时按索引字典序。"""
    d = np.array([[0.0, 0.1, 0.9, 0.95],
                  [0.1, 0.0, 0.92, 0.9],
                  [0.9, 0.92, 0.0, 0.05],
                  [0.95, 0.9, 0.05, 0.0]])
    groups = _agglomerative_groups(d, 2)
    flat = sorted(int(x) for g in groups for x in g)
    assert flat == [0, 1, 2, 3], groups                        # 无遗漏、无重叠
    assert [sorted(g.tolist()) for g in groups] == [[0, 1], [2, 3]]
    assert [g.tolist() for g in groups] == [g.tolist() for g in _agglomerative_groups(d.copy(), 2)]
    assert len(_agglomerative_groups(d, 1)) == 1               # k=1 -> 全部并成一组
    assert len(_agglomerative_groups(d, 4)) == 4               # k=n -> 每资产一组
    assert len(_agglomerative_groups(d, 99)) == 4              # k 越界 -> 裁到 n（每资产一组）
    assert len(_agglomerative_groups(d, 0)) == 1               # k<1 -> 裁到 1
    bad = np.full((3, 3), np.nan)
    assert sorted(int(x) for g in _agglomerative_groups(bad, 2) for x in g) == [0, 1, 2]


def test_signal_ramp_deadband_and_slope():
    z = np.array([0.0, 0.4, 0.5, 1.0, 1.5, 3.0, -1.0, np.nan, np.inf])
    out = _signal_ramp(z, exit_z=0.5, entry_z=1.5)
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0    # 死区内不交易
    assert np.isclose(out[3], 0.5)                              # 线性斜坡
    assert np.isclose(out[4], 1.0) and np.isclose(out[5], 1.0)   # 满强度并截断
    assert np.isclose(out[6], -0.5)                             # 方向跟随 z 的符号
    assert out[7] == 0.0 and out[8] == 0.0                      # 非有限值 -> 不下注
    assert np.abs(out).max() <= 1.0


def test_dollar_neutral_enforces_constraints():
    idx = _bdays(4)
    raw = pd.DataFrame([[0.8, -0.2, 0.0], [3.0, -1.0, 0.0], [np.nan, 2.0, 0.0], [0.0, 0.0, 0.0]],
                       index=idx, columns=["A0", "A1", "A2"])
    d = MarketData(prices=pd.DataFrame(100.0, index=idx, columns=["A0", "A1", "A2"]))
    w = _dollar_neutral(raw, d, budget=1.0)
    assert np.isfinite(w.values).all()
    assert np.abs(w.sum(axis=1)).max() <= SUM_EPS               # 行和被压到 ≈ 0
    assert w.abs().sum(axis=1).max() <= 1.0 + EPS               # 毛敞口被裁到 budget
    assert (w.iloc[3].values == 0.0).all()                      # 单腿行 -> 空仓
    assert np.isclose(w.abs().sum(axis=1).iloc[1], 1.0, atol=1e-12)   # 超预算行被整体缩放
    assert np.isclose(w.iloc[0, 0], -w.iloc[0, 1], atol=1e-12)  # 去均值后两腿等幅


def test_rebalance_grid_is_anchored_at_start():
    assert _rebalance_grid(10, 2, 3) == [2, 5, 8]
    assert _rebalance_grid(10, 12, 3) == []                    # 起点越界 -> 空网格
    assert _rebalance_grid(10, 0, 0) == list(range(10))         # freq<=1 -> 逐期
    # 网格只依赖样本长度与起点，不因未来数据而移动（防未来的结构性保证）
    assert _rebalance_grid(100, 9, 10)[:5] == _rebalance_grid(1000, 9, 10)[:5]


def test_corr_matrix_is_bounded_and_degenerate_safe():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((80, 4))
    X[:, 1] = X[:, 0] * 2.0                                    # 完全相关
    X[:, 3] = 5.0                                              # 常数列 -> 退化
    C = _corr_matrix(X)
    assert np.allclose(C, C.T, atol=0.0)
    assert np.allclose(np.diag(C), 1.0, atol=0.0)
    assert np.isfinite(C).all() and np.abs(C).max() <= 1.0 + 1e-12
    assert np.isclose(C[0, 1], 1.0, atol=1e-12)
    assert (C[3, :3] == 0.0).all() and (C[:3, 3] == 0.0).all()  # 退化列相关系数记 0


def test_strategy_params_are_isolated_between_instances():
    """修改一个实例的 params 不得影响其它实例（类属性字典不可被就地污染）。"""
    a, b = CointPairsPortfolioStrategy(), CointPairsPortfolioStrategy()
    a.params = dict(type(a).params, max_pairs=1)
    assert b.params["max_pairs"] == type(b).params["max_pairs"]
    assert a.params["max_pairs"] == 1
