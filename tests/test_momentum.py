"""momentum 渠道测试：形状/列对齐/有限值、long_only≥0、每行和≤1+eps、确定性，
并含行为断言（强势资产获更高权重、上涨序列被持有）。全部离线、确定。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.momentum import (
    TsMomentumStrategy, XsMomentumStrategy, High52wStrategy, DualMomentumStrategy,
)

ALL_CLASSES = [TsMomentumStrategy, XsMomentumStrategy, High52wStrategy, DualMomentumStrategy]
EPS = 1e-9


def _mk(prices_dict, n):
    """用给定价格列构造 MarketData（工作日索引）。"""
    idx = pd.bdate_range("2020-01-01", periods=n)
    return MarketData(prices=pd.DataFrame(prices_dict, index=idx))


# ---------------- 元信息与无参构造 ----------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_arg_construct_and_meta_complete(cls):
    s = cls()                                   # 必须能无参构造
    assert s.channel == "momentum"
    assert isinstance(s.name, str) and s.name
    assert s.long_only is True
    for field in ("description", "hypothesis", "source"):
        assert getattr(s, field).strip(), field
    assert isinstance(s.params, dict)


# ---------------- 通用约束：形状/对齐/有限/非负/行和/确定性 ----------------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_shape_alignment_finite_and_constraints(cls):
    data = make_synthetic_universe(n_assets=6, n_days=500, seed=7)
    s = cls()
    w = s.generate_weights(data)
    w2 = s.generate_weights(data)               # 再次调用验证确定性

    assert isinstance(w, pd.DataFrame)
    assert w.shape == (len(data.dates), len(data.symbols))
    assert list(w.columns) == data.symbols      # 列对齐
    assert w.index.equals(data.dates)           # 索引对齐
    assert np.isfinite(w.to_numpy()).all()      # 有限值

    if s.long_only:
        assert (w.to_numpy() >= 0).all()        # 非负
    assert w.abs().to_numpy().sum(axis=1).max() <= 1.0 + EPS   # 每行绝对值和 ≤ 1

    pd.testing.assert_frame_equal(w, w2)        # 确定性


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_single_asset_row_cap(cls):
    """单资产极端场景也不得越界（行和≤1、非负、有限）。"""
    n = 300
    data = _mk({"A": np.linspace(100, 260, n)}, n)
    w = cls().generate_weights(data)
    assert w.shape == (n, 1)
    assert np.isfinite(w.to_numpy()).all()
    assert (w.to_numpy() >= 0).all()
    assert w.abs().to_numpy().sum(axis=1).max() <= 1.0 + EPS


# ---------------- 行为断言 ----------------

def test_ts_momentum_holds_uptrend_flat_downtrend():
    """持续上涨序列被持有；持续下跌序列空仓。"""
    n = 200
    data = _mk({"A": np.linspace(100, 200, n),     # 上涨
                "B": np.linspace(200, 100, n)}, n)  # 下跌
    w = TsMomentumStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last["A"] > 0            # 上涨 -> 持有
    assert last["B"] == 0.0         # 下跌 -> 空仓


def test_xs_momentum_prefers_strong_asset():
    """构造明显强于其它的资产，截面动量应给它更高权重。"""
    n = 400
    data = _mk({"S": np.linspace(100, 400, n),     # 强势上涨
                "W1": np.linspace(100, 90, n),     # 弱势下跌
                "W2": np.linspace(100, 80, n)}, n)  # 更弱下跌
    w = XsMomentumStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last["S"] > 0
    assert last["S"] > last["W1"]   # 强势资产权重更高
    assert last["S"] > last["W2"]
    assert last.sum() <= 1.0 + EPS


def test_dual_momentum_prefers_strong_and_filters_negative():
    """双动量：强势资产获更高权重，且绝对动量过滤掉下行的弱势资产。"""
    n = 400
    data = _mk({"S": np.linspace(100, 400, n),
                "W1": np.linspace(100, 90, n),
                "W2": np.linspace(100, 80, n)}, n)
    w = DualMomentumStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last["S"] > 0
    assert last["S"] > last["W1"]
    assert last["S"] > last["W2"]
    # 绝对动量过滤：下行资产（收益<0）不应被持有
    assert last["W1"] == 0.0 and last["W2"] == 0.0


def test_high_52w_holds_near_high_not_crashed():
    """接近滚动新高者持有；冲高后大幅回落者空仓。"""
    n = 300
    rising = np.linspace(100, 300, n)                                    # 一路上行、末期即新高
    crashed = np.concatenate([np.linspace(100, 300, 150),                # 先冲高
                              np.linspace(300, 120, 150)])               # 后重挫
    data = _mk({"R": rising, "C": crashed}, n)
    w = High52wStrategy().generate_weights(data)
    last = w.iloc[-1]
    assert last["R"] > 0                 # 接近新高 -> 持有
    assert last["C"] == 0.0              # 远低于高点 -> 空仓
    assert last["R"] > last["C"]


# ---------------- 集成：可被 discover 自动发现且唯一 ----------------

def test_discover_finds_four_unique_momentum_strategies():
    import kairos_strategies as ks
    names = sorted(s.name for s in ks.discover() if s.channel == "momentum")
    assert names == sorted(["ts_momentum", "xs_momentum", "high_52w", "dual_momentum"])
    assert len(names) == len(set(names))   # 名称唯一
