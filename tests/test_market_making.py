"""market_making 渠道测试：契约（形状/列对齐/有限值/预算）、确定性、防未来、行为断言。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.market_making import (
    AvellanedaStoikovProxyStrategy,
    GridMmDailyStrategy,
    InventorySkewMmStrategy,
    LiquidityProvisionLsStrategy,
)

MM_CLASSES = [
    AvellanedaStoikovProxyStrategy,
    InventorySkewMmStrategy,
    LiquidityProvisionLsStrategy,
    GridMmDailyStrategy,
]

EXPECTED_NAMES = {
    "avellaneda_stoikov_proxy",
    "inventory_skew_mm",
    "liquidity_provision_ls",
    "grid_mm_daily",
}

EPS = 1e-9


def _synth(prices_dict, volumes_dict=None):
    prices = pd.DataFrame(prices_dict)
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    volumes = None
    if volumes_dict is not None:
        volumes = pd.DataFrame(volumes_dict, index=prices.index)
    return MarketData(prices=prices, volumes=volumes)


def _divergent_universe(n_flat: int = 30, n_move: int = 15, rate: float = 0.01):
    """前 n_flat 天平价（参考价预热），随后 DOWN 指数下行 / UP 指数上行。

    末段收盘价持续低于（DOWN）/ 高于（UP）滚动参考价，用于验证库存方向。
    """
    t = np.arange(n_move) + 1
    flat = np.full(n_flat, 100.0)
    down = np.concatenate([flat, 100.0 * np.exp(-rate * t)])
    up = np.concatenate([flat, 100.0 * np.exp(rate * t)])
    return _synth({"DOWN": down, "UP": up})


# ---------------------------- 通用契约 ----------------------------

@pytest.mark.parametrize("cls", MM_CLASSES)
def test_shape_columns_finite(cls):
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=5)
    w = cls().generate_weights(data)
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == data.symbols
    assert len(w) == len(data.dates)
    assert w.index.equals(data.prices.index)
    assert np.isfinite(w.to_numpy(dtype="float64")).all()


@pytest.mark.parametrize("cls", MM_CLASSES)
def test_row_abs_sum_le_one(cls):
    data = make_synthetic_universe(n_assets=7, n_days=300, seed=9)
    w = cls().generate_weights(data)
    v = w.to_numpy(dtype="float64")
    assert np.abs(v).sum(axis=1).max() <= 1.0 + EPS
    assert (np.abs(v) <= 1.0 + EPS).all()          # 单资产权重也 ∈ [-1, 1]


@pytest.mark.parametrize("cls", MM_CLASSES)
def test_deterministic(cls):
    data = make_synthetic_universe(n_assets=5, n_days=250, seed=3)
    a = cls().generate_weights(data)
    b = cls().generate_weights(data)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", MM_CLASSES)
def test_meta(cls):
    s = cls()
    m = s.meta()
    assert m["channel"] == "market_making"
    assert m["universe"] in ("timing", "cross_section")
    assert s.long_only is False                    # 库存型/中性策略均可空
    for k in ("name", "description", "hypothesis", "source", "params"):
        assert m[k]


def test_names_unique_and_expected():
    names = [cls().name for cls in MM_CLASSES]
    assert set(names) == EXPECTED_NAMES
    assert len(names) == len(set(names)) == 4


def test_constructible_without_args():
    for cls in MM_CLASSES:
        assert isinstance(cls(), cls)


# ---------------------------- 美元中性 ----------------------------

@pytest.mark.parametrize("seed", [1, 2])
def test_liquidity_provision_ls_row_sum_zero(seed):
    """liquidity_provision_ls 逐行去均值：行和 ≈ 0（美元中性）且毛敞口 ≤ 1。"""
    data = make_synthetic_universe(n_assets=6, n_days=300, seed=seed)
    w = LiquidityProvisionLsStrategy().generate_weights(data)
    v = w.to_numpy(dtype="float64")
    assert np.abs(v.sum(axis=1)).max() <= 1e-9
    assert np.abs(v).sum(axis=1).max() <= 1.0 + EPS


# ---------------------------- 防未来函数 ----------------------------

@pytest.mark.parametrize("cls", MM_CLASSES)
def test_no_lookahead_tamper_future(cls):
    """篡改 t0 之后的价格，t0 及之前的权重必须逐位不变（尾部窗口 + 时序递推）。"""
    data = make_synthetic_universe(n_assets=5, n_days=300, seed=7)
    t0 = 220
    w_full = cls().generate_weights(data)

    tampered = data.prices.copy()
    n_after = len(tampered) - (t0 + 1)
    factor = (1.37 + 0.11 * np.sin(np.arange(n_after)))[:, None]   # 确定性篡改，恒正
    tampered.iloc[t0 + 1:] = tampered.iloc[t0 + 1:].to_numpy(dtype="float64") * factor
    assert (tampered.to_numpy(dtype="float64") > 0).all()
    data2 = MarketData(prices=tampered, volumes=data.volumes,
                       periods_per_year=data.periods_per_year, name="tampered")
    w_tam = cls().generate_weights(data2)

    a = w_full.iloc[: t0 + 1].to_numpy(dtype="float64")
    b = w_tam.iloc[: t0 + 1].to_numpy(dtype="float64")
    assert a.shape == b.shape
    assert np.array_equal(a, b)                    # 逐位精确相等（强于 allclose）


# ---------------------------- 行为断言 ----------------------------

def test_avellaneda_stoikov_proxy_inventory_direction():
    """价格低于参考价（reservation price）→ 建立多库存；高于 → 减库存/转空。"""
    w = AvellanedaStoikovProxyStrategy().generate_weights(_divergent_universe())
    last = w.iloc[-1]
    assert last["DOWN"] > 0.25                     # 深度低于参考价 → 多库存（≈ +0.5）
    assert last["UP"] < -0.25                      # 深度高于参考价 → 空库存（≈ −0.5）
    assert last["DOWN"] <= 0.5 + EPS               # 库存有界：|权重| ≤ 1/N


def test_inventory_skew_mm_inventory_direction():
    """低于参考价买入加库存至上限；高于参考价卖出减库存转空至下限。"""
    w = InventorySkewMmStrategy().generate_weights(_divergent_universe())
    last = w.iloc[-1]
    assert last["DOWN"] == pytest.approx(0.5, abs=1e-12)   # q = +cap_steps → +1/N
    assert last["UP"] == pytest.approx(-0.5, abs=1e-12)    # q = −cap_steps → −1/N


def test_inventory_skew_mm_quote_skew_prefers_selling_when_long():
    """急跌建满多库存后横盘：带偏斜的报价比无偏斜更早触发卖出（库存向 0 回归）。

    横盘段参考价收敛到现价：无偏斜时 ask = ref×1.01 永不被触及（库存钉在 +cap），
    有偏斜时多库存把 ask 压到 ref×(1+band−skew·band·q/cap) = ref（q=+cap），
    现价触及 ask 触发卖出，库存下降一档 —— 即偏斜使多头库存更倾向卖出。
    """
    px = np.concatenate([np.full(30, 100.0),               # 预热横盘
                         np.linspace(100.0, 90.0, 13)[1:],  # 12 天急跌 → 建满多库存
                         np.full(30, 90.0)])                # 横盘：参考价收敛到 90
    data = _synth({"X": px})
    skewed = InventorySkewMmStrategy()
    w_skew = skewed.generate_weights(data)["X"].iloc[-1]
    flat = InventorySkewMmStrategy()
    flat.params = {**InventorySkewMmStrategy.params, "skew": 0.0}   # 对照：无偏斜
    w_flat = flat.generate_weights(data)["X"].iloc[-1]
    assert w_flat > 0.9                            # 无偏斜：库存钉在 +cap（权重 = 1）
    assert 0.0 < w_skew < w_flat - 0.1             # 有偏斜：多库存触发卖出，库存回落


def test_grid_mm_daily_inventory_direction():
    """连续下穿网格档 → 多库存（权重 > 0）；连续上穿 → 空库存（权重 < 0）。"""
    w = GridMmDailyStrategy().generate_weights(_divergent_universe())
    last = w.iloc[-1]
    assert last["DOWN"] > 0.25                     # 下穿多档 → 多库存
    assert last["UP"] < -0.25                      # 上穿多档 → 转空（区别于 long-only 网格）
    assert np.abs(last).max() <= 0.5 + EPS         # 净库存有界 [-cap, cap]


def test_liquidity_provision_ls_longs_crash_shorts_surge():
    """短期急跌资产做多（提供买方对手盘）、短期急涨资产做空，且美元中性。"""
    n = 60
    crash = np.full(n + 3, 100.0)
    crash[-3:] = [93.0, 86.5, 81.3]                # 最近 3 天急跌
    surge = np.full(n + 3, 100.0)
    surge[-3:] = [107.0, 114.5, 121.4]             # 最近 3 天急涨
    prices = {"CRASH": crash, "SURGE": surge}
    for i in range(4):
        prices[f"FLAT{i}"] = np.full(n + 3, 100.0)
    w = LiquidityProvisionLsStrategy().generate_weights(_synth(prices))
    last = w.iloc[-1]
    assert last.idxmax() == "CRASH"
    assert last.idxmin() == "SURGE"
    assert last["CRASH"] > 0.3                     # 急跌 → 做多
    assert last["SURGE"] < -0.3                    # 急涨 → 做空
    assert abs(last.sum()) <= 1e-9                 # 行和 ≈ 0
    assert np.abs(last).sum() <= 1.0 + EPS         # 绝对值和 ≤ 1


def test_vwap_reference_used_when_volumes_present():
    """成交量存在时参考价用滚动 VWAP：单边下行/上行中库存方向仍正确且契约不破。"""
    n = 120
    t = np.arange(n)
    down = 100.0 * np.exp(-0.01 * t)
    up = 100.0 * np.exp(0.01 * t)
    vols = {"DOWN": np.full(n, 1e6), "UP": np.full(n, 1e6)}
    data = _synth({"DOWN": down, "UP": up}, vols)
    for cls in (AvellanedaStoikovProxyStrategy, InventorySkewMmStrategy, GridMmDailyStrategy):
        w = cls().generate_weights(data)
        v = w.to_numpy(dtype="float64")
        assert np.isfinite(v).all()
        assert w["DOWN"].iloc[-1] > 0.2            # 低于滚动 VWAP → 多库存
        assert w["UP"].iloc[-1] < -0.2             # 高于滚动 VWAP → 空库存
        assert np.abs(v).sum(axis=1).max() <= 1.0 + EPS
