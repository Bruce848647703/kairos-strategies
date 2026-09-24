"""defensive 渠道测试。

覆盖：元信息齐全（中文 description/hypothesis/source、防御异象成因）与无参构造、
形状/列对齐/index 对齐/有限值、权重界限（``betting_against_beta`` 逐行和 ≈ 0 且
两腿各 ±0.5；long_only 三者 ≥ 0，``low_beta_timing``/``defensive_quality`` 逐行和
≈ 1，``drawdown_averse`` 逐行和 ≤ 1；全部每行绝对值和 ≤ 1+eps）、单资产上限
（water-filling）、确定性（同数据/新实例逐位一致）、预热期回退（等权/空仓）、
退化 universe 边界（单资产/超短历史/两资产）、依赖白名单（只用 numpy/pandas）、
全局命名唯一（discover）、引擎可回测，以及两组**防未来断言**：
  * 篡改 t 之后的价格，t 及之前的权重**逐位不变**（bitwise）；
  * 截断样本重算，前缀权重逐位不变；
和**行为断言**：
  * betting_against_beta：对共同市场因子载荷低（β≈0.3）的资产 -> 做多且权重最大，
    载荷高（β≈1.7）的资产 -> 做空；两腿按 beta 分离、腿内权重 ∝ 1/β；净 beta < 0；
  * low_beta_timing：低 beta 资产权重高于等权、高 beta 资产低于等权，组合 beta < 1；
  * defensive_quality / drawdown_averse：贴近滚动高点且低波的 CALM 权重最大，
    深回撤的 CRASH 被剔除（权重 = 0）；全市场深回撤时 drawdown_averse 退向全现金。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, discover, make_synthetic_universe
from kairos_strategies.channels import defensive as dmod
from kairos_strategies.channels.defensive import (
    BettingAgainstBetaStrategy,
    DefensiveQualityStrategy,
    DrawdownAverseStrategy,
    LowBetaTimingStrategy,
)

STRATEGIES = [BettingAgainstBetaStrategy, LowBetaTimingStrategy,
              DefensiveQualityStrategy, DrawdownAverseStrategy]
EXPECTED_NAMES = {"betting_against_beta", "low_beta_timing",
                  "defensive_quality", "drawdown_averse"}
LONG_ONLY = {LowBetaTimingStrategy, DefensiveQualityStrategy, DrawdownAverseStrategy}
EPS = 1e-9            # 值域 / 绝对值和容差
SUM_EPS = 1e-9        # 逐行权重和容差
LEG_BUDGET = 0.5      # BAB 单腿预算（两腿绝对值之和 = 1）
PREFIX_ATOL = 0.0     # 防未来断言：要求逐位（bitwise）一致
# 预热行数（首个非空仓行之前的整行空仓数），由窗口参数决定，已实测核对
WARMUP_ROWS = {
    BettingAgainstBetaStrategy: 59,    # = beta_window - 1（凑不出两腿前空仓）
    DefensiveQualityStrategy: 125,     # = ma_window - 1（慢均线是三者中最长窗口）
    DrawdownAverseStrategy: 62,        # = dd_window - 1（滚动高点窗口）
}
FULL_INVESTED = {LowBetaTimingStrategy, DefensiveQualityStrategy}   # 逐行和 ≈ 1
W_CAP_DEFAULT = 0.40  # low_beta_timing 的单资产权重上限


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


# 低 beta / 高 beta 对同一共同市场因子 f 的载荷（构造真值，用于行为断言）
LOADINGS = {"LOW": 0.30, "HIGH": 1.70, "M0": 1.00, "M1": 1.00, "M2": 1.00, "M3": 1.00}


def _beta_universe(n: int = 320, seed: int = 21) -> MarketData:
    """LOW 对共同因子载荷 0.30（低 beta）、HIGH 载荷 1.70（高 beta）、4 只 M 载荷 1.00。

    等权市场代理 = 各资产收益的截面均值，其因子载荷恰为 1.00（6 只载荷均值），
    故各资产相对「市场」的真实 beta ≈ 其载荷，行为断言可用构造真值直接核对。
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n) * 0.010                      # 共同市场因子
    cols = {}
    for sym, load in LOADINGS.items():
        r = load * f + rng.standard_normal(n) * 0.0015      # 因子暴露 + 少量特质噪声
        cols[sym] = 100.0 * np.cumprod(1.0 + r)
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _drawdown_universe(n: int = 260, seed: int = 7) -> MarketData:
    """CALM 平滑上涨、贴近自身滚动高点且低波；CRASH 后段暴跌形成深回撤且高波。"""
    rng = np.random.default_rng(seed)
    k = int(n * 0.62)
    cols = {
        "CALM": 100.0 * np.exp(np.linspace(0.0, 0.30, n)
                               + np.cumsum(rng.standard_normal(n) * 0.0006)),
        "CRASH": 100.0 * np.exp(np.concatenate([np.linspace(0.0, 0.25, k),
                                                np.linspace(0.25, -0.35, n - k)])
                                + np.cumsum(rng.standard_normal(n) * 0.008)),
    }
    for i in range(4):
        cols[f"M{i}"] = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.010))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2020-01-02")))


def _all_deep_drawdown_universe(n: int = 200, seed: int = 3) -> MarketData:
    """6 只资产在后段**同时**回撤 25%~35%（系统性风险），用于验证整体退向现金。"""
    rng = np.random.default_rng(seed)
    k = int(n * 0.6)
    cols = {}
    for i in range(6):
        top = 0.30 + 0.02 * i
        path = np.concatenate([np.linspace(0.0, top, k), np.linspace(top, -0.10, n - k)])
        cols[f"C{i}"] = 100.0 * np.exp(path + np.cumsum(rng.standard_normal(n) * 0.004))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2021-01-04")))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=520, seed=11)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "defensive"
        assert s.long_only is (cls in LONG_ONLY), s.name
        assert s.universe in ("timing", "cross_section"), s.name
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 30, (s.name, key)
        # hypothesis 必须说明防御异象成因（杠杆约束/彩票偏好/回撤不对称等）
        assert any(kw in meta["hypothesis"] for kw in
                   ("杠杆", "彩票", "回撤", "beta", "波动")), (s.name, "hypothesis 缺成因")
        assert "失效" in meta["hypothesis"], (s.name, "hypothesis 缺失效条件")
        assert isinstance(meta["params"], dict) and meta["params"]
    assert names == EXPECTED_NAMES
    assert dmod.CHANNEL == "defensive"
    assert set(dmod.STRATEGY_NAMES) == EXPECTED_NAMES


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


def test_weight_bounds(synth):
    """long_only ≥ 0；可多空 ∈ [-1,1]；所有策略每行绝对值和 ≤ 1+eps（不加杠杆）。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
        assert w.values.max() <= 1.0 + EPS, cls.name
        if cls in LONG_ONLY:
            assert w.values.min() >= -EPS, cls.name
        else:
            assert w.values.min() >= -1.0 - EPS, cls.name
            assert w.values.max() > 0.0 and w.values.min() < 0.0, cls.name


def test_bab_row_sum_zero_and_leg_budgets(synth):
    """BAB 硬约束：逐行和 ≈ 0（美元中性），非空仓行两腿各 ≈ ±0.5、毛敞口 = 1。"""
    w = BettingAgainstBetaStrategy().generate_weights(synth)
    row_sum = w.sum(axis=1)
    assert np.abs(row_sum).max() <= SUM_EPS, float(np.abs(row_sum).max())
    assert np.allclose(row_sum.values, 0.0, atol=SUM_EPS, rtol=0.0)
    act = w.abs().sum(axis=1) > 0
    assert int(act.sum()) >= 400, int(act.sum())
    longs = w.clip(lower=0.0).sum(axis=1)[act]
    shorts = w.clip(upper=0.0).sum(axis=1)[act]
    assert np.allclose(longs.values, LEG_BUDGET, atol=SUM_EPS)
    assert np.allclose(shorts.values, -LEG_BUDGET, atol=SUM_EPS)
    assert np.allclose(w.abs().sum(axis=1)[act].values, 2.0 * LEG_BUDGET, atol=SUM_EPS)
    assert (w[act].max(axis=1) > 0).all() and (w[act].min(axis=1) < 0).all()


def test_long_only_row_sums(synth):
    """满仓者逐行和 ≈ 1；drawdown_averse 逐行和 ≤ 1（差额为现金）。"""
    for cls in LONG_ONLY:
        w = cls().generate_weights(synth)
        total = w.sum(axis=1)
        assert total.max() <= 1.0 + EPS, cls.name
        assert total.min() >= -EPS, cls.name
        if cls is DrawdownAverseStrategy:
            assert total.iloc[WARMUP_ROWS[cls]:].max() <= 1.0 + EPS
        elif cls is LowBetaTimingStrategy:
            # 满仓配置：含预热期在内每一行都 ≈ 1（预热期回退等权 1/N）
            assert np.allclose(total.values, 1.0, atol=SUM_EPS), cls.name
        else:  # defensive_quality：预热期空仓，其后满仓
            warm = WARMUP_ROWS[cls]
            assert np.allclose(total.values[:warm], 0.0, atol=SUM_EPS), cls.name
            assert np.allclose(total.values[warm:], 1.0, atol=SUM_EPS), cls.name


def test_single_asset_weight_cap_respected():
    """low_beta_timing 的 water-filling 上限：单资产 ≤ max(w_cap, 1/N)。"""
    s = LowBetaTimingStrategy()
    for d, n in ((_beta_universe(), 6), (make_synthetic_universe(n_assets=8, n_days=400, seed=3), 8),
                 (MarketData(prices=_beta_universe().prices[["LOW", "HIGH"]]), 2)):
        w = s.generate_weights(d)
        cap = max(W_CAP_DEFAULT, 1.0 / n)
        assert w.values.max() <= cap + EPS, (n, w.values.max(), cap)
        assert np.allclose(w.sum(axis=1).values, 1.0, atol=SUM_EPS)


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)


def test_warmup_fallback(synth):
    """预热期：BAB/defensive_quality/drawdown_averse 空仓，low_beta_timing 回退等权。"""
    n = len(synth.symbols)
    for cls, warm in WARMUP_ROWS.items():
        w = cls().generate_weights(synth)
        assert (w.iloc[:warm].values == 0.0).all(), (cls.name, warm)
        assert w.abs().sum(axis=1).iloc[warm] > 0.0, (cls.name, warm)
    lw = LowBetaTimingStrategy().generate_weights(synth)
    warm = int(LowBetaTimingStrategy.params["beta_window"]) - 1
    assert np.allclose(lw.iloc[:warm].values, 1.0 / n, atol=1e-12)   # 等权满仓回退
    assert not np.allclose(lw.iloc[warm].values, 1.0 / n, atol=1e-12)  # 预热结束即倾斜
    assert np.allclose(lw.sum(axis=1).values, 1.0, atol=SUM_EPS)


# ------------------------------------------------------------- 防未来函数

def test_no_lookahead_tampered_future_prices(synth):
    """篡改 t 之后的价格（换成完全不同的随机路径），t 及之前的权重必须**逐位不变**。"""
    t = 300
    rng = np.random.default_rng(2024)
    tampered = synth.prices.copy()
    shape = tampered.iloc[t + 1:].shape
    tampered.iloc[t + 1:] = 100.0 * np.exp(rng.standard_normal(shape) * 0.60)
    d2 = MarketData(prices=tampered, volumes=synth.volumes,
                    periods_per_year=synth.periods_per_year, name="tampered")
    assert not np.allclose(d2.prices.values[t + 1:], synth.prices.values[t + 1:])
    for cls in STRATEGIES:
        s = cls()
        base = s.generate_weights(synth).values[:t + 1]
        alt = s.generate_weights(d2).values[:t + 1]
        assert np.array_equal(base, alt), cls.name          # bitwise 相同


def test_no_lookahead_scaled_and_reversed_future(synth):
    """另一种篡改（未来价格整体放大 + 上下翻转）同样不得影响 t 及之前的权重。"""
    t = 250
    p = synth.prices.copy()
    p.iloc[t + 1:] = np.flipud(p.iloc[t + 1:].values) * 3.7
    d2 = MarketData(prices=p, periods_per_year=synth.periods_per_year, name="scaled")
    for cls in STRATEGIES:
        s = cls()
        assert np.array_equal(s.generate_weights(synth).values[:t + 1],
                              s.generate_weights(d2).values[:t + 1]), cls.name


def test_no_lookahead_prefix_invariance(synth):
    """截断样本重算：前缀权重与全样本逐位一致（所有统计量都是 trailing / 当期截面）。"""
    m = 400
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.allclose(full, part, atol=PREFIX_ATOL, rtol=0.0), cls.name
        assert np.array_equal(full, part), cls.name


def test_no_full_sample_statistics_in_source():
    """源码级防未来：不得出现全样本/向后填充类操作（expanding、bfill、shift(-n)）。"""
    src = inspect.getsource(dmod)
    for bad in ("expanding(", "bfill", "backfill", "shift(-", ".iloc[-1]"):
        assert bad not in src, bad


# ------------------------------------------------------------- 边界与集成

def test_degenerate_universes_are_safe():
    """单资产 / 超短历史 / 两资产 都必须安全返回（合法权重、无 NaN/inf）。"""
    base = make_synthetic_universe(n_assets=3, n_days=400, seed=5)
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    for cls in STRATEGIES:
        s = cls()
        for d in (one, short, two):
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, cls.name
            assert list(w.columns) == d.symbols and w.index.equals(d.dates), cls.name
            assert np.isfinite(w.values).all(), cls.name
            assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
            assert w.values.max() <= 1.0 + EPS, cls.name
        # 单资产：BAB 凑不出两腿 -> 全 0；low_beta_timing 独享满仓 1.0；
        # drawdown_averse 视单资产为整个组合（预热期后按自身回撤给敞口，可为 0）
        w1 = s.generate_weights(one)
        if cls is LowBetaTimingStrategy:
            assert np.allclose(w1.values, 1.0, atol=SUM_EPS), cls.name
        elif cls is DrawdownAverseStrategy:
            warm = WARMUP_ROWS[cls]
            assert (w1.iloc[:warm].values == 0.0).all(), cls.name
            assert w1.values.min() >= -EPS, cls.name
            assert w1.sum(axis=1).max() <= 1.0 + EPS, cls.name
        else:
            assert (w1.values == 0.0).all(), cls.name
        # 超短历史（10 天 < 所有窗口）：BAB/quality/drawdown 空仓，low_beta 等权满仓
        ws = s.generate_weights(short)
        if cls is LowBetaTimingStrategy:
            assert np.allclose(ws.sum(axis=1).values, 1.0, atol=SUM_EPS), cls.name
            assert np.allclose(ws.values, 1.0 / 3.0, atol=1e-12), cls.name
        else:
            assert (ws.values == 0.0).all(), cls.name
        # 两资产：BAB 一多一空各 0.5；long_only 者逐行和 ≈ 1 或 ≤ 1
        w2 = s.generate_weights(two)
        if cls is BettingAgainstBetaStrategy:
            assert np.abs(w2.sum(axis=1)).max() <= SUM_EPS, cls.name
            act = w2.abs().sum(axis=1) > 0
            assert np.allclose(w2.clip(lower=0.0).sum(axis=1)[act].values,
                               LEG_BUDGET, atol=SUM_EPS), cls.name
        elif cls is LowBetaTimingStrategy:
            assert np.allclose(w2.sum(axis=1).values, 1.0, atol=SUM_EPS), cls.name
            assert w2.values.max() <= 0.5 + EPS, cls.name      # cap = max(0.4, 1/2)
        else:
            assert w2.sum(axis=1).max() <= 1.0 + EPS, cls.name
            assert w2.values.min() >= -EPS, cls.name


def test_only_numpy_pandas_dependencies():
    """依赖白名单：本渠道只用 numpy/pandas（beta/回撤/配权工具全部自实现）。"""
    tree = ast.parse(inspect.getsource(dmod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "pandas", "typing", "__future__"}, imported
    for bad in ("statsmodels", "sklearn", "scipy"):
        assert bad not in dmod.__dict__, bad
    # 不得 import 其它渠道的私有函数
    src = inspect.getsource(dmod)
    for bad in ("channels.technical", "channels.factor", "channels.long_short",
                "channels.regime", "from . import"):
        assert bad not in src, bad


def test_discover_registers_channel_with_unique_names():
    """全局注册：4 个策略被 discover 找到、channel 分组正确、名字全局唯一不重复。"""
    all_strats = discover()
    names = [s.name for s in all_strats]
    assert len(names) == len(set(names)), "存在重名策略"
    assert EXPECTED_NAMES <= set(names)
    grouped = [s for s in all_strats if s.channel == "defensive"]
    assert {s.name for s in grouped} == EXPECTED_NAMES


def test_engine_accepts_weights(synth):
    """权重可直接进引擎：收益/换手有限、换手不虚增（≤ 2）。"""
    bt = Backtester()
    for cls in STRATEGIES:
        res = bt.run(synth, cls().generate_weights(synth))
        assert np.isfinite(res.returns.values).all(), cls.name
        assert np.isfinite(res.turnover.values).all(), cls.name
        assert res.turnover.max() <= 2.0 + 1e-6, (cls.name, res.turnover.max())
        assert np.isfinite(res.metrics["sharpe"]), cls.name


# ------------------------------------------------------------------ 行为断言

def test_betting_against_beta_longs_low_beta_shorts_high_beta():
    """对共同因子载荷 0.30 的 LOW -> 做多且权重最大；载荷 1.70 的 HIGH -> 做空。"""
    s = BettingAgainstBetaStrategy()
    d = _beta_universe()
    w = s.generate_weights(d)
    last = w.iloc[-1]
    load = pd.Series(LOADINGS).reindex(d.symbols)
    assert last["LOW"] > 0.0                                   # 低 beta -> 多头腿
    assert last["HIGH"] < 0.0                                  # 高 beta -> 空头腿
    assert last["LOW"] == last.max()                           # 最低 beta 拿最大权重（1/β）
    assert int((last > 0).sum()) == 2 and int((last < 0).sum()) == 2   # n=6, k=ceil(6/3)=2
    assert int((last == 0).sum()) == 2                         # 中间分位不下注
    assert np.abs(last.sum()) <= SUM_EPS                       # 美元中性
    assert np.isclose(last.abs().sum(), 1.0, atol=SUM_EPS)     # 毛敞口 1（不加杠杆）
    # 两腿按 beta 严格分离，且腿内权重 ∝ 1/β（beta 越低权重越大 / 空头负得越少）
    beta = dmod._estimated_beta(d, s.params["beta_window"], s.params["shrink"]).iloc[-1]
    longs, shorts = last[last > 0].index, last[last < 0].index
    assert beta.loc[longs].max() < beta.loc[shorts].min()
    lo = beta.loc[longs].sort_values().index
    assert (last.loc[lo].diff().dropna() < 0.0).all()
    sh = beta.loc[shorts].sort_values().index
    assert (last.loc[sh].diff().dropna() > 0.0).all()
    # 净 beta 暴露为负（防御属性）：用构造时的真实因子载荷核对
    net_beta = w.mul(load, axis=1).sum(axis=1)
    act = w.abs().sum(axis=1) > 0
    assert float(net_beta.iloc[-1]) < 0.0
    assert (net_beta[act] < 0.0).all()
    assert float(net_beta[act].max()) < -0.1


def test_low_beta_timing_overweights_low_beta():
    """逆 beta 满仓配置：低 beta 权重高于等权、高 beta 低于等权，组合 beta < 1。"""
    s = LowBetaTimingStrategy()
    d = _beta_universe()
    w = s.generate_weights(d)
    last = w.iloc[-1]
    eq = 1.0 / len(d.symbols)
    load = pd.Series(LOADINGS).reindex(d.symbols)
    assert last["LOW"] > eq > last["HIGH"]                     # 低 beta 加权、高 beta 减权
    assert last["LOW"] == last.max() and last["HIGH"] == last.min()
    assert np.allclose(w.sum(axis=1).values, 1.0, atol=SUM_EPS)  # 满仓
    assert w.values.min() >= -EPS
    # 组合 beta（用构造真值载荷）严格低于等权基准的 1.0，预热期恰为 1.0（等权回退）
    port_beta = w.mul(load, axis=1).sum(axis=1)
    warm = int(s.params["beta_window"]) - 1
    assert np.allclose(port_beta.values[:warm], 1.0, atol=1e-9)
    assert float(port_beta.iloc[warm:].max()) < 1.0 - 0.10
    assert float(port_beta.iloc[-1]) < float(port_beta.iloc[:warm].mean())


def test_defensive_quality_prefers_calm_near_high_over_deep_drawdown():
    """防御质量：贴近滚动高点、低波、慢均线之上的 CALM 权重最大；深回撤 CRASH 被剔除。"""
    s = DefensiveQualityStrategy()
    d = _drawdown_universe()
    w = s.generate_weights(d)
    last = w.iloc[-1]
    assert last["CALM"] > 0.0 and last["CALM"] == last.max()
    assert last["CRASH"] == 0.0                                # 深回撤 -> 不入篮子
    assert int((last > 0).sum()) == 2                          # n=6, top_frac=1/3 -> k=2
    assert np.isclose(last.sum(), 1.0, atol=SUM_EPS)           # 满仓
    assert w.values.min() >= -EPS
    # CRASH 进入深回撤后的整段（后 60 日）都不被选中；CALM 全程被选中
    assert (w["CRASH"].iloc[-60:] == 0.0).all()
    assert (w["CALM"].iloc[-60:] > 0.0).all()
    assert (w["CALM"].iloc[125:] == w.iloc[125:].max(axis=1)).all()   # 每期都是第一重仓


def test_drawdown_averse_drops_deep_drawdown_prefers_near_high():
    """回撤规避：dd ≥ dd_tol 的 CRASH 权重清零；near-high 低波的 CALM 拿最大权重。"""
    s = DrawdownAverseStrategy()
    d = _drawdown_universe()
    w = s.generate_weights(d)
    last = w.iloc[-1]
    dd = dmod._current_drawdown(dmod._panel(d), int(s.params["dd_window"])).iloc[-1]
    assert float(dd["CRASH"]) >= float(s.params["dd_tol"])     # CRASH 确实处于深回撤
    assert float(dd["CALM"]) < 0.02                            # CALM 贴近自身滚动高点
    assert last["CRASH"] == 0.0                                # 深回撤 -> 完全剔除
    assert last["CALM"] == last.max() and last["CALM"] > 0.0
    assert 0.0 < float(last.sum()) <= 1.0 + EPS                # 行和 ≤ 1（差额是现金）
    assert float(last.sum()) < 1.0 - 0.05                      # 确有防御性现金留存
    assert w.values.min() >= -EPS
    assert (w["CRASH"].iloc[-40:] == 0.0).all()
    # 全市场同时深回撤 -> 组合层健康度缩放把整体压到全现金
    w_all = s.generate_weights(_all_deep_drawdown_universe())
    assert (w_all.iloc[-1].values == 0.0).all()
    assert float(w_all.iloc[-1].sum()) == 0.0
    assert w_all.sum(axis=1).max() <= 1.0 + EPS
