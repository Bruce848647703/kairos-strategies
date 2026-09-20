"""momentum_adv 渠道测试（进阶动量变体）。

覆盖：元信息齐全（中文 description/hypothesis/source）与无参构造、形状/列对齐/
index 对齐/有限值、权重界限（long_only 权重 ≥ 0 且行和 ∈ [0,1]；多空行和 ≈ 0）、
每行绝对值和 ≤ 1+eps、确定性（同数据/新实例逐位一致）、预热期空仓、退化 universe
边界、依赖白名单（只用 numpy/pandas）、全局命名唯一（discover）、引擎可回测，
**防未来函数（篡改 t 之后价格，t 及之前权重逐位不变）**，以及**行为断言**：
  * frog_in_pan：平滑小步上涨 vs 单日跳空上涨（同涨幅）-> 更偏好平滑者；
  * vol_managed_momentum：低波段总敞口 > 高波段总敞口（高波降杠杆）；
  * momentum_spread_timing：一强多弱 -> 做多强者/做空弱者且行和≈0；价差崩溃时平仓避险；
  * group_momentum：一强多弱 -> 做多最强组/做空最弱组且行和≈0。
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import Backtester, MarketData, discover, make_synthetic_universe
from kairos_strategies.channels.momentum_adv import (
    FrogInPanStrategy,
    GroupMomentumStrategy,
    MomentumSpreadTimingStrategy,
    VolManagedMomentumStrategy,
)

ALL_CLASSES = [FrogInPanStrategy, VolManagedMomentumStrategy,
               MomentumSpreadTimingStrategy, GroupMomentumStrategy]
LONG_ONLY = [FrogInPanStrategy, VolManagedMomentumStrategy]
NEUTRAL = [MomentumSpreadTimingStrategy, GroupMomentumStrategy]
EXPECTED_NAMES = {"frog_in_pan", "vol_managed_momentum",
                  "momentum_spread_timing", "group_momentum"}
EPS = 1e-9            # 绝对值和 / 值域容差
SUM_EPS = 1e-9        # 每行权重和（美元中性）容差
PREFIX_ATOL = 1e-12   # 防未来：前缀不变性容差（浮点噪声级别）


# ------------------------------------------------------------------ 数据构造

def _bdays(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _smooth_vs_jump(n: int = 120, gain: float = 0.30) -> MarketData:
    """两只累计涨幅相同的资产：SMOOTH 许多小涨日平滑累积；JUMP 平时不动、末日单日跳空。"""
    smooth = 100.0 * np.exp(np.linspace(0.0, np.log(1.0 + gain), n))
    jump = np.full(n, 100.0)
    jump[-1] = 100.0 * (1.0 + gain)
    return MarketData(prices=pd.DataFrame({"SMOOTH": smooth, "JUMP": jump}, index=_bdays(n)))


def _vol_regime_universe(n: int = 300, split: int = 150, seed: int = 5) -> MarketData:
    """6 只上涨资产：前半段极低波、后半段极高波（共同市场冲击主导，用于检验波动缩放）。"""
    rng = np.random.default_rng(seed)
    sigma = np.concatenate([np.full(split, 0.002), np.full(n - split, 0.05)])
    common = rng.standard_normal(n) * sigma                 # 共同冲击 -> 市场层波动
    cols = {}
    for i in range(6):
        r = 0.001 + common + rng.standard_normal(n) * 0.001  # 正漂移 + 共同 + 微小特质
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _one_strong_many_weak(n: int = 300) -> MarketData:
    """WIN 全程强势上涨，5 只 LOSER 逐级走弱（一强多弱）。"""
    cols = {"WIN": 100.0 * np.exp(np.linspace(0.0, 0.80, n))}
    for j in range(5):
        cols[f"L{j}"] = 100.0 * np.exp(np.linspace(0.0, -0.10 * (j + 1), n))
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


def _spread_crash_universe(n: int = 340, pre: int = 300) -> MarketData:
    """前 ``pre`` 天价差持续走阔（WIN 涨、LOSER 跌）；末段急剧反转（WIN 崩、LOSER 弹）。"""
    win = np.concatenate([np.linspace(0.0, 1.0, pre), np.linspace(1.0, -0.2, n - pre)])
    cols = {"WIN": 100.0 * np.exp(win)}
    for j in range(5):
        lo = np.concatenate([np.linspace(0.0, -0.15 * (j + 1), pre),
                             np.linspace(-0.15 * (j + 1), 0.6 + 0.1 * j, n - pre)])
        cols[f"L{j}"] = 100.0 * np.exp(lo)
    return MarketData(prices=pd.DataFrame(cols, index=_bdays(n)))


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=6, n_days=520, seed=11)


# ------------------------------------------------------------------ 元信息与契约

def test_meta_complete_and_no_arg_construction():
    names = set()
    for cls in ALL_CLASSES:
        s = cls()                                             # 必须无参可构造
        meta = s.meta()
        assert s.channel == "momentum_adv"
        assert s.universe == "cross_section"
        assert s.name in EXPECTED_NAMES and s.name not in names
        names.add(s.name)
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"]
    assert names == EXPECTED_NAMES
    # long_only 标记必须与实现一致
    for cls in LONG_ONLY:
        assert cls().long_only is True, cls.name
    for cls in NEUTRAL:
        assert cls().long_only is False, cls.name


def test_shape_columns_index_alignment_and_finite(synth):
    for cls in ALL_CLASSES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == (len(synth.dates), len(synth.symbols)), cls.name
        assert list(w.columns) == synth.symbols
        assert w.index.equals(synth.dates)
        assert np.isfinite(w.to_numpy()).all(), cls.name


def test_weight_bounds(synth):
    """long_only 权重 ≥ 0 且行和 ∈ [0,1]；多空行和 ≈ 0；所有策略每行绝对值和 ≤ 1+eps。"""
    for cls in ALL_CLASSES:
        w = cls().generate_weights(synth)
        row_sum = w.sum(axis=1)
        abs_sum = w.abs().sum(axis=1)
        assert abs_sum.max() <= 1.0 + EPS, cls.name
        assert w.to_numpy().max() <= 1.0 + EPS, cls.name
        assert w.to_numpy().min() >= -1.0 - EPS, cls.name
        if cls in LONG_ONLY:
            assert w.to_numpy().min() >= -EPS, cls.name           # 非负
            assert row_sum.min() >= -EPS and row_sum.max() <= 1.0 + EPS, cls.name
        else:
            assert np.abs(row_sum).max() <= SUM_EPS, cls.name     # 美元中性
            assert w.to_numpy().min() < 0.0, f"{cls.name} 从未做空"
            assert w.to_numpy().max() > 0.0, f"{cls.name} 从未做多"


def test_warmup_rows_flat(synth):
    """信息不足的预热早期必须整行空仓。"""
    for cls in ALL_CLASSES:
        w = cls().generate_weights(synth)
        assert (w.iloc[:5].to_numpy() == 0.0).all(), cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in ALL_CLASSES:
        s = cls()
        w1 = s.generate_weights(synth)
        w2 = s.generate_weights(synth)
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.to_numpy(), cls().generate_weights(synth).to_numpy())


# ------------------------------------------------------------------ 防未来函数

def test_no_lookahead_tamper_future(synth):
    """篡改 t 之后的价格，t 及之前每一行权重必须逐位不变（防未来函数）。"""
    data = make_synthetic_universe(n_assets=6, n_days=400, seed=11)
    t = 300                                                   # 已过所有策略预热期
    p2 = data.prices.copy()
    p2.iloc[t + 1:] = p2.iloc[t + 1:] * 2.5 + 37.0            # 任意篡改未来价格
    data2 = MarketData(prices=p2, volumes=None,
                       periods_per_year=data.periods_per_year, name=data.name)
    for cls in ALL_CLASSES:
        s = cls()
        w_full = s.generate_weights(data).iloc[:t + 1].to_numpy()
        w_tamper = s.generate_weights(data2).iloc[:t + 1].to_numpy()
        assert np.allclose(w_full, w_tamper, atol=PREFIX_ATOL, rtol=0.0), cls.name


def test_no_lookahead_prefix_invariance(synth):
    """截断样本重算，前缀权重与全样本一致（另一角度验证不含未来信息）。"""
    m = 400
    sub = MarketData(prices=synth.prices.iloc[:m],
                     volumes=None if synth.volumes is None else synth.volumes.iloc[:m],
                     periods_per_year=synth.periods_per_year, name=synth.name)
    for cls in ALL_CLASSES:
        s = cls()
        full = s.generate_weights(synth).to_numpy()[:m]
        part = s.generate_weights(sub).to_numpy()
        assert np.allclose(full, part, atol=PREFIX_ATOL, rtol=0.0), cls.name


# ------------------------------------------------------------------ 退化 universe 安全性

def test_degenerate_universes_are_safe():
    base = make_synthetic_universe(n_assets=3, n_days=400, seed=5)
    one = MarketData(prices=base.prices[["A0"]])
    short = MarketData(prices=base.prices.iloc[:10])
    two = MarketData(prices=base.prices[["A0", "A1"]])
    for cls in ALL_CLASSES:
        s = cls()
        for d in (one, short, two):
            w = s.generate_weights(d)
            assert w.shape == d.prices.shape, cls.name
            assert np.isfinite(w.to_numpy()).all(), cls.name
            assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
            if cls in LONG_ONLY:
                assert w.to_numpy().min() >= -EPS, cls.name
            else:
                assert np.abs(w.sum(axis=1)).max() <= SUM_EPS, cls.name
        # 单资产凑不出多空两腿 / 选不出截面 -> 全 0（中性）或有界（只做多）
        w1 = s.generate_weights(one)
        if cls in NEUTRAL:
            assert (w1.to_numpy() == 0.0).all(), cls.name


# ------------------------------------------------------------------ 依赖白名单 & 全局注册

def test_only_numpy_pandas_dependencies():
    """本渠道必须只用 numpy/pandas（因子/排序/分组工具全部自实现）。"""
    import kairos_strategies.channels.momentum_adv as mod
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
    all_strats = discover()
    names = [s.name for s in all_strats]
    assert len(names) == len(set(names)), "存在重名策略"
    assert EXPECTED_NAMES <= set(names)
    grouped = {s.name for s in all_strats if s.channel == "momentum_adv"}
    assert grouped == EXPECTED_NAMES


def test_engine_accepts_weights(synth):
    bt = Backtester()
    for cls in ALL_CLASSES:
        w = cls().generate_weights(synth)
        res = bt.run(synth, w)
        assert np.isfinite(res.returns.to_numpy()).all(), cls.name
        assert np.isfinite(res.turnover.to_numpy()).all(), cls.name
        # 美元中性、毛敞口≤1 的组合换手硬上界≈2（多头/空头组整体互换），留出漂移余量
        assert res.turnover.max() <= 2.5, (cls.name, res.turnover.max())


# ------------------------------------------------------------------ 行为断言

def test_frog_in_pan_prefers_smooth_over_jump():
    """同样累计涨幅：平滑小步上涨者被做多，单日跳空上涨者被规避（frog 分更高）。"""
    w = FrogInPanStrategy().generate_weights(_smooth_vs_jump())
    last = w.iloc[-1]
    assert last["SMOOTH"] > 0.0                     # 平滑上涨 -> 做多
    assert last["JUMP"] == 0.0                      # 跳空动量 -> 规避
    assert last["SMOOTH"] > last["JUMP"]
    assert last.sum() <= 1.0 + EPS


def test_vol_managed_momentum_delevers_in_high_vol():
    """高波段总敞口显著低于低波段（波动管理，控制动量崩溃）。"""
    w = VolManagedMomentumStrategy().generate_weights(_vol_regime_universe())
    exposure = w.sum(axis=1)                        # long_only，行和即总敞口
    low_seg = exposure.iloc[130:150].mean()         # 低波段（缩放上限满仓）
    high_seg = exposure.iloc[250:300].mean()        # 高波段（降杠杆）
    assert low_seg > high_seg, (low_seg, high_seg)
    assert high_seg < 0.6 * low_seg, (low_seg, high_seg)
    assert exposure.max() <= 1.0 + EPS              # 不加杠杆


def test_momentum_spread_timing_longs_strong_shorts_weak():
    """一强多弱且价差走强：做多最强者、做空最弱者，行和≈0（gate 开仓）。"""
    w = MomentumSpreadTimingStrategy().generate_weights(_one_strong_many_weak())
    last = w.iloc[-1]
    assert last.abs().sum() > 0.0                   # 价差动量为正 -> 已开仓
    assert last["WIN"] > 0.0 and last["WIN"] == last.max()
    assert last["L4"] < 0.0                         # 最弱者被做空
    assert last.min() < 0.0
    assert abs(last.sum()) <= SUM_EPS               # 美元中性
    assert last.abs().sum() <= 1.0 + EPS


def test_momentum_spread_timing_flattens_on_spread_crash():
    """价差崩溃（赢家转弱、输家反弹）时择时开关平仓避险：反转段绝大多数行空仓。"""
    w = MomentumSpreadTimingStrategy().generate_weights(_spread_crash_universe(pre=300))
    active = (w.abs().sum(axis=1) > 0)
    pre_open = active.iloc[260:295]                 # 价差走阔期：应持仓
    crash_closed = active.iloc[305:335]             # 价差崩溃期：应平仓
    assert pre_open.all(), f"价差走阔期未持仓: {int(pre_open.sum())}/{len(pre_open)}"
    assert not crash_closed.any(), f"崩溃期未平仓: {int(crash_closed.sum())}/{len(crash_closed)}"


def test_group_momentum_longs_strongest_group_shorts_weakest():
    """一强多弱：最强者所在组被做多、最弱者所在组被做空，行和≈0。"""
    w = GroupMomentumStrategy().generate_weights(_one_strong_many_weak())
    last = w.iloc[-1]
    assert last["WIN"] > 0.0 and last["WIN"] == last.max()   # 最强 -> 多头组
    assert last["L3"] < 0.0 and last["L4"] < 0.0             # 最弱 -> 空头组
    assert last.min() < 0.0
    assert abs(last.sum()) <= SUM_EPS                        # 美元中性
    assert last.abs().sum() <= 1.0 + EPS
