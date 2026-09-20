"""statarb 渠道测试。

覆盖：形状/列对齐/有限值、每行绝对值和 <= 1+eps、**每行权重和 ≈ 0（美元中性）**、
值域 [-1,1]、确定性、预热期空仓、退化 universe 边界、防未来函数（前缀不变性）、
依赖白名单（只用 numpy/pandas，不得引入 statsmodels/sklearn/scipy）、元信息与无参构造，
以及四条**行为断言**：
  * coint_pairs：构造高度协整（价差平稳）的两腿，价差走阔 -> 做空贵腿/做多便宜腿（符号相反），
    价差收窄 -> 反向；并用「第三只纯随机游走资产」验证配对筛选真的挑了协整对。
  * eof_stat_arb：构造一个明显偏离主成分（共同因子）的资产 -> 权重与该偏离方向相反（均值回归）。
  * xs_zscore_reversion：构造截面明显偏弱/偏强的资产 -> 弱者做多、强者做空。
  * basket_neutral：构造相对更贵的篮子 -> 做空贵篮子、做多便宜篮子，两篮子预算各半且行和为 0。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, make_synthetic_universe
from kairos_strategies.channels.statarb import (
    BasketNeutralStrategy,
    CointPairsStrategy,
    EofStatArbStrategy,
    XsZscoreReversionStrategy,
)

STRATEGIES = [CointPairsStrategy, EofStatArbStrategy,
              XsZscoreReversionStrategy, BasketNeutralStrategy]
EXPECTED_NAMES = {"coint_pairs", "eof_stat_arb", "xs_zscore_reversion", "basket_neutral"}
EPS = 1e-9                 # 绝对值和 / 值域容差
SUM_EPS = 1e-9             # 每行权重和（美元中性）容差
PREFIX_ATOL = 1e-12        # 前缀不变性容差（矩阵乘法的 BLAS 舍入噪声）


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _coint_universe(shock: float = 0.20, n: int = 270, tail: int = 3,
                    third: bool = False) -> MarketData:
    """高度协整的两腿：logy = 0.3 + 1.1*logx + 平稳残差（正弦 + 小噪声）。

    ``shock > 0`` 让 A1 在末段相对 A0 **走阔**（A1 变贵），``shock < 0`` 则收窄。
    n=270 是刻意选的：末段冲击（最后 ``tail`` 天）落在最后一次再平衡的估计窗口**之外**，
    因此选对/估 β 用的是干净样本，而残差 z-score 能看到冲击 —— 这正是实盘的时序。
    ``third=True`` 再加一只独立的纯随机游走 A2（与 A0/A1 不协整），用于检验配对筛选。
    """
    rng = np.random.default_rng(7)
    logx = np.log(100.0) + np.cumsum(rng.standard_normal(n) * 0.01)
    resid = 0.02 * np.sin(np.arange(n) * 0.35) + rng.standard_normal(n) * 0.004  # 平稳
    logy = 0.3 + 1.1 * logx + resid
    if shock:
        logy[-tail:] += shock                                  # 末段价差跳升/跳降
    cols = {"A0": np.exp(logx), "A1": np.exp(logy)}
    if third:
        logz = np.log(80.0) + np.cumsum(rng.standard_normal(n) * 0.015)  # 独立随机游走
        cols["A2"] = np.exp(logz)
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _factor_universe(dev: float = 1.0, n: int = 360, n_assets: int = 6,
                     dev_col: int = 3, tail: int = 4, per_day: float = 0.02) -> MarketData:
    """单一强共同因子（主成分）+ 弱特质噪声；末段让 ``A{dev_col}`` 明显偏离主成分。

    ``dev > 0``：该资产末段被额外推高（实际价格高于因子隐含的公允价格）；``dev < 0`` 反之。
    """
    rng = np.random.default_rng(3)
    factor = rng.standard_normal(n) * 0.02                     # 共同因子（PC1 = 市场）
    logp = np.zeros((n, n_assets))
    for i in range(n_assets):
        logp[:, i] = np.log(50.0 + 10.0 * i) + np.cumsum(factor + rng.standard_normal(n) * 0.004)
    if dev:
        logp[-tail:, dev_col] += dev * per_day * np.arange(1, tail + 1)
    cols = [f"A{i}" for i in range(n_assets)]
    return MarketData(prices=pd.DataFrame(np.exp(logp), index=_bdays(n), columns=cols))


def _xs_universe(weak: bool = True, n: int = 120, n_assets: int = 5,
                 col: int = 2, ramp: int = 12, per_day: float = 0.01) -> MarketData:
    """截面里让 ``A{col}`` 近端持续偏弱（weak=True）或偏强，其余资产窄幅震荡。"""
    rng = np.random.default_rng(5)
    logp = np.zeros((n, n_assets))
    for i in range(n_assets):
        logp[:, i] = np.log(100.0) + np.cumsum(rng.standard_normal(n) * 0.003)
    sign = -1.0 if weak else 1.0
    logp[-(ramp + 1):, col] += sign * per_day * np.arange(ramp + 1)
    cols = [f"A{i}" for i in range(n_assets)]
    return MarketData(prices=pd.DataFrame(np.exp(logp), index=_bdays(n, "2022-01-03"), columns=cols))


def _basket_universe(cheap_basket: str = "high", n: int = 300) -> MarketData:
    """A0/A1 = 低波篮子，A2/A3 = 高波篮子；``cheap_basket`` 指定末段被推高（变贵）的那个篮子。"""
    t = np.arange(n)
    lo0 = np.log(100.0) + np.cumsum(0.0020 * np.sin(t * 0.50))
    lo1 = np.log(100.0) + np.cumsum(0.0022 * np.cos(t * 0.47))
    hi2 = np.log(100.0) + np.cumsum(0.0200 * np.sin(t * 0.90))
    hi3 = np.log(100.0) + np.cumsum(0.0210 * np.cos(t * 0.85))
    bump = 0.010 * np.arange(1, 31)
    if cheap_basket == "high":
        hi2[-30:] += bump
        hi3[-30:] += 0.9 * bump
    else:
        lo0[-30:] += bump
        lo1[-30:] += 0.9 * bump
    cols = {"A0": np.exp(lo0), "A1": np.exp(lo1), "A2": np.exp(hi2), "A3": np.exp(hi3)}
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n, "2021-01-01")))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=420, seed=11)


# ------------------------------------------------------------- 契约与数值健全

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in STRATEGIES:
        s = cls()                                            # 必须无参可构造
        meta = s.meta()
        assert s.channel == "statarb"
        assert s.universe == "cross_section"
        assert s.long_only is False                          # 市场中性 / 多空
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 10, (s.name, key)
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


def test_row_abs_sum_le_one(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name


def test_row_sum_is_zero_dollar_neutral(synth):
    """市场中性硬约束：每一行权重之和都必须 ≈ 0（含空仓行）。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        row_sum = w.sum(axis=1)
        assert np.abs(row_sum).max() <= SUM_EPS, (cls.name, np.abs(row_sum).max())
        assert np.allclose(row_sum.values, 0.0, atol=SUM_EPS, rtol=0.0), cls.name


def test_weights_within_unit_range(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() >= -1.0 - EPS, cls.name
        assert w.values.max() <= 1.0 + EPS, cls.name


def test_long_short_both_signs_appear(synth):
    """可多空策略必须真的同时产生多头与空头，且非空仓行足够多（不是几乎不交易）。"""
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.values.min() < 0.0, f"{cls.name} 从未做空"
        assert w.values.max() > 0.0, f"{cls.name} 从未做多"
        active = int((w.abs().sum(axis=1) > 0).sum())
        assert active >= 50, (cls.name, active)
        # 每个非空仓行都同时含多头与空头（否则就不是中性组合）
        rows = w[w.abs().sum(axis=1) > 0]
        assert (rows.max(axis=1) > 0).all() and (rows.min(axis=1) < 0).all(), cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.values, w2.values)
        # 不同实例（同参数）也必须一致：策略无可变状态
        assert np.array_equal(w1.values, cls().generate_weights(synth).values)


def test_warmup_rows_are_flat(synth):
    """信息不足的预热期必须整行空仓（不下注），且各策略的预热长度符合其窗口参数。"""
    cases = {
        CointPairsStrategy: max(CointPairsStrategy.params["est_window"],
                               CointPairsStrategy.params["z_window"]) - 1,
        EofStatArbStrategy: EofStatArbStrategy.params["window"] - 1,
        XsZscoreReversionStrategy: (XsZscoreReversionStrategy.params["lookback"]
                                    + XsZscoreReversionStrategy.params["smooth"] - 1),
        BasketNeutralStrategy: max(BasketNeutralStrategy.params["z_window"],
                                   BasketNeutralStrategy.params["vol_window"]) - 1,
    }
    for cls, warm in cases.items():
        w = cls().generate_weights(synth)
        assert (w.iloc[:warm].values == 0.0).all(), (cls.name, warm)


def test_no_lookahead_prefix_invariance(synth):
    """防未来函数：截断样本重算，前缀权重必须与全样本一致（到浮点噪声级别）。

    再平衡网格锚定在样本起点，所有估计都用尾部窗口，故新增未来数据不会改写历史权重。
    eof 的载荷投影是矩阵乘法，截断后最后一个 block 的行数变化会带来 ~1e-18 的 BLAS
    舍入差异，因此对最后一行只要求机器精度级一致，其余行要求逐位相同。
    """
    m = 300
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).values[:m]
        part = s.generate_weights(sub).values
        assert np.allclose(full, part, atol=PREFIX_ATOL, rtol=0.0), cls.name
        head_full, head_part = full[:m - 1], part[:m - 1]
        assert np.allclose(head_full, head_part, atol=0.0, rtol=0.0), cls.name


def test_only_numpy_pandas_dependencies():
    """禁止 statsmodels / sklearn / scipy：协整与 PCA 必须是自研 numpy 实现。"""
    import kairos_strategies.channels.statarb as mod
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


def test_degenerate_universes_are_safe():
    """单资产 / 历史过短 / 两资产 等退化输入都必须安全返回（全 0 或合法中性权重）。"""
    base = make_synthetic_universe(n_assets=3, n_days=400, seed=5)
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    for cls in STRATEGIES:
        s = cls()
        for d in (one, short):
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, cls.name
            assert (w.values == 0.0).all(), (cls.name, d.prices.shape)   # 无法构成多空 -> 空仓
        w2 = s.generate_weights(two)
        assert w2.shape == two.prices.shape
        assert np.isfinite(w2.values).all()
        assert w2.abs().sum(axis=1).max() <= 1.0 + EPS
        assert np.abs(w2.sum(axis=1)).max() <= SUM_EPS


def test_engine_accepts_dollar_neutral_weights(synth):
    """行和为 0 的权重可直接进引擎，换手有限且不虚增（引擎按 1+r_p 归一）。"""
    bt = Backtester()
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        res = bt.run(synth, w)
        assert np.isfinite(res.returns.values).all(), cls.name
        assert np.isfinite(res.turnover.values).all(), cls.name
        assert res.turnover.max() <= 2.0 + 1e-6, (cls.name, res.turnover.max())
        assert res.turnover.mean() < 1.0, cls.name


# ------------------------------------------------------------------ 行为断言

def test_coint_pairs_shorts_widening_spread_and_longs_narrowing():
    """价差走阔（A1 相对 A0 变贵）-> 做空 A1 / 做多 A0；价差收窄 -> 反向。"""
    s = CointPairsStrategy()
    wide = s.generate_weights(_coint_universe(shock=+0.20))
    narrow = s.generate_weights(_coint_universe(shock=-0.20))
    for w, leg_long, leg_short in ((wide, "A0", "A1"), (narrow, "A1", "A0")):
        last = w.iloc[-1]
        assert last[leg_short] < 0.0 < last[leg_long]          # 两腿符号相反
        assert np.isclose(last[leg_long], -last[leg_short], atol=EPS)   # 等预算反向
        assert np.sign(last[leg_long]) * np.sign(last[leg_short]) < 0
    assert wide["A1"].iloc[-1] < 0.0 and narrow["A1"].iloc[-1] > 0.0
    assert np.abs(wide.sum(axis=1)).max() <= SUM_EPS           # 全程美元中性
    assert wide.abs().sum(axis=1).max() <= 1.0 + EPS


def test_coint_pairs_only_trades_the_selected_pair():
    """任意时刻只有被选中的两条腿有权重，其余资产恒为 0。"""
    s = CointPairsStrategy()
    w = s.generate_weights(_coint_universe(shock=+0.20, third=True))
    per_row = (w.values != 0.0).sum(axis=1)
    assert set(np.unique(per_row)) <= {0, 2}, np.unique(per_row)  # 空仓或恰好两腿


def test_coint_pairs_prefers_cointegrated_legs_over_random_walk():
    """配对筛选（半衰期代理 ADF）必须挑中真协整对，而不是独立随机游走。"""
    s = CointPairsStrategy()
    w = s.generate_weights(_coint_universe(shock=+0.20, third=True))
    assert (w["A2"].values == 0.0).all()                       # 随机游走腿从未被选中
    assert (w[["A0", "A1"]].values != 0.0).any()               # 协整对被选中并交易


def test_coint_pairs_trades_much_less_on_pure_random_walks():
    """配对筛选（ADF t 统计量 + 半衰期门槛）必须真的在起作用：

    在「两条腿都是独立随机游走」（本就不协整）的 universe 上，交易活跃度应显著低于
    真协整 universe；否则说明筛选形同虚设、在做伪回归的价差。
    """
    s = CointPairsStrategy()
    rng = np.random.default_rng(17)
    n = 400
    logp = np.cumsum(rng.standard_normal((n, 2)) * 0.02, axis=0) + np.log(100.0)
    prices = pd.DataFrame(np.exp(logp), index=_bdays(n), columns=["A0", "A1"])
    w_rw = s.generate_weights(MarketData(prices=prices))
    w_ci = s.generate_weights(_coint_universe(shock=0.0))
    act_rw = (w_rw.abs().sum(axis=1) > 0).mean()
    act_ci = (w_ci.abs().sum(axis=1) > 0).mean()
    assert act_rw < 0.5 * act_ci, (act_rw, act_ci)
    for w in (w_rw, w_ci):                                     # 无论如何都保持中性/无杠杆
        assert np.abs(w.sum(axis=1)).max() <= SUM_EPS
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS


def test_eof_stat_arb_shorts_asset_deviating_above_factor_fair_value():
    """明显偏离主成分（被推高）的资产 -> 负权重；被压低 -> 正权重（均值回归）。"""
    s = EofStatArbStrategy()
    up = s.generate_weights(_factor_universe(dev=+1.0))
    down = s.generate_weights(_factor_universe(dev=-1.0))
    assert up["A3"].iloc[-1] < 0.0                             # 高于公允价格 -> 做空
    assert down["A3"].iloc[-1] > 0.0                           # 低于公允价格 -> 做多
    # 偏离资产应是被下注最重的一批（绝对权重不低于该行的中位数）
    for w in (up, down):
        last = w.iloc[-1]
        assert abs(last["A3"]) >= np.median(np.abs(last.values))
    # 组合仍是美元中性、毛敞口 <= 1，且空头篮子的总权重被多头抵消
    for w in (up, down):
        assert np.abs(w.sum(axis=1)).max() <= SUM_EPS
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS
        longs, shorts = w.clip(lower=0.0).sum(axis=1), (-w).clip(lower=0.0).sum(axis=1)
        act = w.abs().sum(axis=1) > 0
        assert np.allclose(longs[act].values, shorts[act].values, atol=SUM_EPS)


def test_eof_stat_arb_uses_only_leading_components():
    """投影矩阵必须是对称幂等的真投影，秩 = 保留的主成分个数（m 超过 N 时退化为单位阵）。"""
    r = np.random.default_rng(0).standard_normal((200, 6)) * 0.01
    proj = EofStatArbStrategy._projector(r, 2)
    assert proj is not None and proj.shape == (6, 6)
    assert np.allclose(proj, proj.T, atol=1e-12)               # 对称
    assert np.allclose(proj @ proj, proj, atol=1e-12)          # 幂等（真投影）
    assert np.isclose(np.trace(proj), 2.0, atol=1e-9)          # 秩 = m = 2
    proj1 = EofStatArbStrategy._projector(r, 1)
    assert np.isclose(np.trace(proj1), 1.0, atol=1e-9)
    # 保留全部成分 -> P = I -> 残差恒为 0 -> 组合空仓（说明信号确实来自「被剔除的成分」）
    assert np.allclose(EofStatArbStrategy._projector(r, 99), np.eye(6), atol=1e-9)
    assert EofStatArbStrategy._projector(np.zeros((200, 6)), 2) is None   # 退化窗口 -> None


def test_xs_zscore_reversion_longs_weak_and_shorts_strong():
    """截面偏弱（近期跌）-> 做多；截面偏强（近期涨）-> 做空；其余资产反向配平。"""
    s = XsZscoreReversionStrategy()
    weak = s.generate_weights(_xs_universe(weak=True))
    strong = s.generate_weights(_xs_universe(weak=False))
    others = [c for c in weak.columns if c != "A2"]
    assert weak["A2"].iloc[-1] > 0.0                           # 弱 -> 多
    assert (weak[others].iloc[-1] < 0.0).all()                 # 其余 -> 空（配平）
    assert strong["A2"].iloc[-1] < 0.0                         # 强 -> 空
    assert (strong[others].iloc[-1] > 0.0).all()
    for w in (weak, strong):
        assert np.abs(w.sum(axis=1)).max() <= SUM_EPS
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS
        assert abs(w["A2"].iloc[-1]) == w.abs().iloc[-1].max()  # 偏离资产拿到最大权重


def test_basket_neutral_shorts_expensive_basket():
    """高波篮子被推高（相对更贵）-> 做空高波篮子、做多低波篮子；两篮子预算各半。"""
    s = BasketNeutralStrategy()
    w = s.generate_weights(_basket_universe(cheap_basket="high"))
    last = w.iloc[-1]
    assert last["A0"] > 0.0 and last["A1"] > 0.0               # 低波篮子（便宜）做多
    assert last["A2"] < 0.0 and last["A3"] < 0.0               # 高波篮子（贵）做空
    assert np.isclose(last["A0"], last["A1"], atol=EPS)        # 篮子内部等权
    assert np.isclose(last["A2"], last["A3"], atol=EPS)
    gross = w.abs().sum(axis=1)
    long_total = w[["A0", "A1"]].sum(axis=1)
    short_total = -w[["A2", "A3"]].sum(axis=1)
    act = gross > 0
    assert np.allclose(long_total[act], short_total[act], atol=SUM_EPS)      # 行和 ≈ 0
    assert np.allclose(long_total[act].abs(), gross[act] / 2.0, atol=SUM_EPS)  # 两篮子各占一半预算
    assert gross.max() <= 1.0 + EPS
    # 反向情形：低波篮子被推高 -> 变成做空低波篮子
    w2 = s.generate_weights(_basket_universe(cheap_basket="low")).iloc[-1]
    assert w2["A0"] < 0.0 and w2["A2"] > 0.0


def test_basket_neutral_partitions_every_asset_exactly_once():
    """篮子划分必须是全 universe 的一个二分（无遗漏、无重叠），两篮子皆非空。"""
    s = BasketNeutralStrategy()
    d = _basket_universe()
    vols = d.rolling_vol(s.params["vol_window"]).iloc[-1].to_numpy(dtype="float64")
    a, b = s._split_baskets(vols, d.n_assets)
    assert sorted(a.tolist() + b.tolist()) == list(range(d.n_assets))
    assert a.size > 0 and b.size > 0 and not (set(a) & set(b))
    assert vols[a].max() <= vols[b].min()                      # A = 低波，B = 高波
    small_a, small_b = s._split_baskets(np.full(3, np.nan), 3)  # 信息缺失 -> 索引奇偶兜底
    assert sorted(small_a.tolist() + small_b.tolist()) == [0, 1, 2]
