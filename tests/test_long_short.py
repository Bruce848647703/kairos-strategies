"""long_short 渠道测试。

覆盖：元信息齐全（中文 description/hypothesis/source）与无参构造、形状/列对齐/
index 对齐/有限值、**每行权重和 ≈ 0（美元中性）**、每行绝对值和 ≤ 1+eps、值域
[-1,1]、两腿预算各 ≈ ±0.5、多空两腿同时存在、确定性（同数据/新实例逐位一致）、
预热期空仓、退化 universe 边界（单资产/超短历史/两资产）、防未来函数（前缀不变性）、
依赖白名单（只用 numpy/pandas）、全局命名唯一（discover），以及**行为断言**：
  * xs_momentum_ls / residual_momentum_ls：持续显著跑赢的资产 -> 做多，持续跑输 -> 做空；
  * reversal_ls：近期大跌资产 -> 做多，近期大涨资产 -> 做空；
  * lowvol_ls：低波资产 -> 做多，高波资产 -> 做空；
  * quality_ls：平滑上升趋势（高效率+低波）-> 做多，剧烈震荡（低效率+高波）-> 做空。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.long_short import (
    CrossSectionMomentumLongShortStrategy,
    LowVolLongShortStrategy,
    QualityLongShortStrategy,
    ResidualMomentumLongShortStrategy,
    ReversalLongShortStrategy,
)

STRATEGIES = [CrossSectionMomentumLongShortStrategy, ReversalLongShortStrategy,
              QualityLongShortStrategy, ResidualMomentumLongShortStrategy,
              LowVolLongShortStrategy]
EXPECTED_NAMES = {"xs_momentum_ls", "reversal_ls", "quality_ls",
                  "residual_momentum_ls", "lowvol_ls"}
EPS = 1e-9            # 绝对值和 / 值域容差
SUM_EPS = 1e-9        # 每行权重和（美元中性）容差
LEG_BUDGET = 0.5      # 单腿预算（两腿绝对值之和 = 1）
PREFIX_ATOL = 1e-12   # 前缀不变性容差（浮点噪声级别）
# 各策略预热行数（首个有权重行之前的整行空仓数），由窗口参数决定，已实测核对
WARMUP_ROWS = {
    CrossSectionMomentumLongShortStrategy: 252,   # = lookback
    ReversalLongShortStrategy: 5,                 # = lookback
    QualityLongShortStrategy: 60,                 # = window（效率比）
    ResidualMomentumLongShortStrategy: 118,       # = reg_window + score_window - 2
    LowVolLongShortStrategy: 41,                  # = vol_window - 1
}


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _trend_ls_universe(n: int = 320, seed: int = 21) -> MarketData:
    """WIN 全程显著跑赢、LOSE 全程显著跑输，其余 4 只为小幅随机游走（截面陪跑）。"""
    rng = np.random.default_rng(seed)
    cols = {
        "WIN": 100.0 * np.exp(np.linspace(0.0, 0.60, n)
                              + np.cumsum(rng.standard_normal(n) * 0.002)),
        "LOSE": 100.0 * np.exp(np.linspace(0.0, -0.50, n)
                               + np.cumsum(rng.standard_normal(n) * 0.002)),
    }
    for i in range(4):
        cols[f"M{i}"] = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.005))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _reversal_universe(n: int = 120, tail: int = 5, seed: int = 33) -> MarketData:
    """DROP 最近 ``tail`` 天暴跌 38%、POP 暴涨 45%，其余 5 只温和漂移（幅度 ≤ 3%）。"""
    rng = np.random.default_rng(seed)
    cols = {}
    for i in range(5):
        drift = np.linspace(0.0, 0.02 * (1.0 if i % 2 else -1.0), n)
        cols[f"M{i}"] = 100.0 * np.exp(drift + np.cumsum(rng.standard_normal(n) * 0.002))
    flat = np.full(n - tail, 100.0)
    cols["DROP"] = np.concatenate([flat, np.linspace(100.0, 62.0, tail + 1)[1:]])
    cols["POP"] = np.concatenate([flat, np.linspace(100.0, 145.0, tail + 1)[1:]])
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2022-01-03")))


def _vol_universe(n: int = 260, seed: int = 42) -> MarketData:
    """6 只资产波动阶梯分明：LOW/LOW2 极低波，MID 居中，HIGH/HIGH2 极高波。"""
    rng = np.random.default_rng(seed)
    sigmas = {"LOW": 0.0008, "LOW2": 0.0012, "MID1": 0.008,
              "MID2": 0.010, "HIGH": 0.035, "HIGH2": 0.045}
    cols = {k: 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * s))
            for k, s in sigmas.items()}
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _quality_universe(n: int = 220, seed: int = 8) -> MarketData:
    """SMOOTH 平滑单边上涨（效率比≈1、波动极低）；CHOPPY 大幅正弦震荡（效率低、波动极高）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {
        "SMOOTH": 100.0 * np.exp(np.linspace(0.0, 0.35, n)
                                 + np.cumsum(rng.standard_normal(n) * 0.0004)),
        "CHOPPY": 100.0 * (1.0 + 0.18 * np.sin(t * 1.05))
                          * np.exp(np.cumsum(rng.standard_normal(n) * 0.006)),
    }
    for i in range(4):
        cols[f"M{i}"] = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.010))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2021-01-01")))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=520, seed=11)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "long_short"
        assert s.universe == "cross_section"
        assert s.long_only is False                          # 多空 alpha
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"]
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


def test_row_sum_is_zero_dollar_neutral(synth):
    """多空硬约束：每一行权重之和都必须 ≈ 0（±1e-9，含空仓行）。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        row_sum = w.sum(axis=1)
        assert np.abs(row_sum).max() <= SUM_EPS, (cls.name, np.abs(row_sum).max())
        assert np.allclose(row_sum.values, 0.0, atol=SUM_EPS, rtol=0.0), cls.name


def test_row_abs_sum_le_one_and_range(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
        assert w.values.min() >= -1.0 - EPS, cls.name
        assert w.values.max() <= 1.0 + EPS, cls.name


def test_two_legs_each_half_budget(synth):
    """非空仓行：多头腿合计 ≈ +0.5、空头腿合计 ≈ -0.5（两腿各自归一到半预算）。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        gross = w.abs().sum(axis=1)
        act = gross > 0
        longs = w.clip(lower=0.0).sum(axis=1)[act]
        shorts = w.clip(upper=0.0).sum(axis=1)[act]
        assert np.allclose(longs.values, LEG_BUDGET, atol=SUM_EPS), cls.name
        assert np.allclose(shorts.values, -LEG_BUDGET, atol=SUM_EPS), cls.name


def test_long_and_short_legs_both_present(synth):
    """可多空策略必须真的同时做多与做空，且每个非空仓行都两腿俱全。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() < 0.0, f"{cls.name} 从未做空"
        assert w.values.max() > 0.0, f"{cls.name} 从未做多"
        rows = w[w.abs().sum(axis=1) > 0]
        assert len(rows) >= 200, (cls.name, len(rows))
        assert (rows.max(axis=1) > 0).all() and (rows.min(axis=1) < 0).all(), cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        # 新实例（同参数）也必须一致：策略无可变状态
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)


def test_warmup_rows_are_flat(synth):
    """信息不足的预热期必须整行空仓；预热结束后立即恢复下注。"""
    for cls, warm in WARMUP_ROWS.items():
        w = cls().generate_weights(synth)
        assert (w.iloc[:warm].values == 0.0).all(), (cls.name, warm)
        assert w.abs().sum(axis=1).iloc[warm] > 0.0, (cls.name, warm)


def test_no_lookahead_prefix_invariance(synth):
    """防未来函数：截断样本重算，前缀权重必须与全样本一致（浮点噪声级别）。

    所有因子都只用滚动尾部窗口 / shift（历史）价格与当期截面排序，
    故加入更多未来数据不会改写历史上任何一行的权重。
    """
    m = 400
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.allclose(full, part, atol=PREFIX_ATOL, rtol=0.0), cls.name


def test_degenerate_universes_are_safe():
    """单资产 / 超短历史 / 两资产 等退化输入都必须安全返回（合法中性权重）。"""
    base = make_synthetic_universe(n_assets=3, n_days=400, seed=5)
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    long_warm = {CrossSectionMomentumLongShortStrategy, QualityLongShortStrategy,
                 ResidualMomentumLongShortStrategy, LowVolLongShortStrategy}
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(one)                          # 单资产凑不出两腿 -> 全 0
        assert w1.shape == one.prices.shape
        assert (w1.values == 0.0).all(), cls.name
        for d in (short, two):
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, cls.name
            assert np.isfinite(w.values).all(), cls.name
            assert np.abs(w.sum(axis=1)).max() <= SUM_EPS, cls.name
            assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
        if cls in long_warm:                                  # 预热窗口 > 10 天 -> 全空仓
            assert (s.generate_weights(short).values == 0.0).all(), cls.name
        w2 = s.generate_weights(two)                          # 两资产：一多一空各 0.5
        per_row = np.unique((w2.values != 0.0).sum(axis=1))
        assert set(per_row.tolist()) <= {0, 2}, (cls.name, per_row)
        act = w2.abs().sum(axis=1) > 0
        if act.any():
            longs = w2.clip(lower=0.0).sum(axis=1)[act]
            assert np.allclose(longs.values, LEG_BUDGET, atol=SUM_EPS), cls.name


def test_only_numpy_pandas_dependencies():
    """依赖白名单：本渠道必须只用 numpy/pandas（因子/排序工具全部自实现）。"""
    import kairos_strategies.channels.long_short as mod
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
    """全局注册：5 个策略被 discover 找到、channel 分组正确、名字全局唯一不重复。"""
    all_strats = discover()
    names = [s.name for s in all_strats]
    assert len(names) == len(set(names)), "存在重名策略"
    assert EXPECTED_NAMES <= set(names)
    grouped = [s for s in all_strats if s.channel == "long_short"]
    assert {s.name for s in grouped} == EXPECTED_NAMES


def test_engine_accepts_neutral_weights(synth):
    """行和为 0 的多空权重可直接进引擎：收益/换手有限且不虚增。"""
    bt = Backtester()
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        res = bt.run(synth, w)
        assert np.isfinite(res.returns.values).all(), cls.name
        assert np.isfinite(res.turnover.values).all(), cls.name
        assert res.turnover.max() <= 2.0 + 1e-6, (cls.name, res.turnover.max())


# ------------------------------------------------------------------ 行为断言

def test_xs_momentum_ls_longs_winner_shorts_loser():
    """持续显著跑赢的资产 -> 做多；持续跑输的资产 -> 做空（12-1 截面动量）。"""
    s = CrossSectionMomentumLongShortStrategy()
    w = s.generate_weights(_trend_ls_universe())
    last = w.iloc[-1]
    assert last["WIN"] > 0.0                                   # 最强 -> 多头腿
    assert last["LOSE"] < 0.0                                  # 最弱 -> 空头腿
    assert np.isclose(last["WIN"], LEG_BUDGET / 2.0, atol=EPS)  # 6 资产 k=2 -> 每腿 0.25
    assert np.isclose(last["LOSE"], -LEG_BUDGET / 2.0, atol=EPS)
    assert abs(w.sum(axis=1)).max() <= SUM_EPS                 # 全程美元中性
    assert last["WIN"] == last.max() and last["LOSE"] == last.min()


def test_residual_momentum_ls_longs_winner_shorts_loser():
    """回归掉等权市场后，特质残差动量最强的资产 -> 做多，最弱 -> 做空。"""
    s = ResidualMomentumLongShortStrategy()
    w = s.generate_weights(_trend_ls_universe())
    last = w.iloc[-1]
    assert last["WIN"] > 0.0
    assert last["LOSE"] < 0.0
    assert last["WIN"] == last.max() and last["LOSE"] == last.min()
    longs = w.clip(lower=0.0).sum(axis=1)
    shorts = w.clip(upper=0.0).sum(axis=1)
    act = w.abs().sum(axis=1) > 0
    assert np.allclose(longs[act].values, -shorts[act].values, atol=SUM_EPS)
    assert np.allclose(longs[act].values, LEG_BUDGET, atol=SUM_EPS)


def test_reversal_ls_longs_recent_crasher_shorts_recent_popper():
    """近期暴跌资产 -> 做多（博反弹），近期暴涨资产 -> 做空（博回吐）。"""
    s = ReversalLongShortStrategy()
    w = s.generate_weights(_reversal_universe())
    last = w.iloc[-1]
    assert last["DROP"] > 0.0                                  # 大跌 -> 多头腿
    assert last["POP"] < 0.0                                   # 大涨 -> 空头腿
    assert last["DROP"] == last.max() and last["POP"] == last.min()
    assert abs(w.sum(axis=1)).max() <= SUM_EPS


def test_lowvol_ls_longs_low_vol_shorts_high_vol():
    """低波分位 -> 做多，高波分位 -> 做空（低波异象的市场中性版）。"""
    s = LowVolLongShortStrategy()
    w = s.generate_weights(_vol_universe())
    last = w.iloc[-1]
    assert last["LOW"] > 0.0 and last["LOW2"] > 0.0            # 两只低波都做多
    assert last["HIGH"] < 0.0 and last["HIGH2"] < 0.0          # 两只高波都做空
    assert np.isclose(last["LOW"], LEG_BUDGET / 2.0, atol=EPS)
    assert np.isclose(last["HIGH2"], -LEG_BUDGET / 2.0, atol=EPS)
    assert last["MID1"] == 0.0 and last["MID2"] == 0.0         # 中间分位不入两腿
    assert last.abs().sum() <= 1.0 + EPS


def test_quality_ls_longs_smooth_uptrend_shorts_choppy():
    """平滑单边上涨（高效率+低波）-> 做多；剧烈震荡（低效率+高波）-> 做空。"""
    s = QualityLongShortStrategy()
    w = s.generate_weights(_quality_universe())
    last = w.iloc[-1]
    assert last["SMOOTH"] > 0.0
    assert last["CHOPPY"] < 0.0
    assert last["SMOOTH"] == last.max() and last["CHOPPY"] == last.min()
    assert abs(w.sum(axis=1)).max() <= SUM_EPS
    assert w.abs().sum(axis=1).max() <= 1.0 + EPS
