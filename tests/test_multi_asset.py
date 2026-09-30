"""multi_asset 多资产配置策略的离线测试（用受控合成价格，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import realdata
from kairos_strategies.base import MarketData
from kairos_strategies.multi_asset import AllWeather, GTAA, SixtyForty

SYMS = realdata.etf_symbols if hasattr(realdata, "etf_symbols") else None
ALL = [s for mem in realdata.ASSET_CLASSES.values() for s in mem]


def _mk(trends, vols, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {}
    for s in ALL:
        cls = realdata.CLASS_OF[s]
        mu = trends.get(cls, 0.0)
        sg = vols.get(cls, 0.15)
        r = mu / 252 + sg / np.sqrt(252) * rng.standard_normal(n)
        cols[s] = 100 * np.exp(np.cumsum(r))
    return MarketData(prices=pd.DataFrame(cols, index=idx), periods_per_year=252, name="test")


def test_all_weather_sums_to_one_and_prefers_low_vol():
    data = _mk(trends={"cn_equity": 0.1, "bond": 0.03, "cash": 0.02},
               vols={"cn_equity": 0.25, "global_equity": 0.22, "commodity": 0.16, "bond": 0.03, "cash": 0.001})
    w = AllWeather(window=40).generate_weights(data)
    tail = w.iloc[60:]
    assert (w.values >= -1e-12).all()
    np.testing.assert_allclose(tail.sum(axis=1).values, 1.0, atol=1e-6)  # 满仓
    bond_w = tail[realdata.ASSET_CLASSES["bond"]].sum(axis=1).mean()
    eq_w = tail[realdata.ASSET_CLASSES["cn_equity"]].sum(axis=1).mean()
    assert bond_w > eq_w            # 低波债券类别权重更高（风险平价）
    # 现金类不参与 all_weather
    assert tail[realdata.ASSET_CLASSES["cash"]].sum(axis=1).mean() == pytest.approx(0.0, abs=1e-9)


def test_gtaa_bounds_and_cash_when_all_negative():
    trends = {c: -0.8 for c in realdata.ASSET_CLASSES}
    trends["cash"] = 0.02                      # 现金为正，作为 retreat
    vols = {c: 0.05 for c in realdata.ASSET_CLASSES}  # 低噪声 -> 动量稳定为负
    down = _mk(trends=trends, vols=vols, n=600)
    w = GTAA(lookback=120, top_k=2, vol_window=40).generate_weights(down)
    tail = w.iloc[200:]
    assert (w.values >= -1e-12).all()
    assert (tail.sum(axis=1).values <= 1.0 + 1e-6).all()
    # 全类别稳定下跌 -> 绝对动量过滤 -> 基本全现金
    cash = realdata.ASSET_CLASSES["cash"]
    assert tail[cash].sum(axis=1).mean() > 0.9


def test_gtaa_holds_strong_class():
    data = _mk(trends={"cn_equity": 0.6, "global_equity": -0.2, "commodity": -0.2, "bond": -0.1, "cash": 0.01},
               vols={c: 0.18 for c in realdata.ASSET_CLASSES}, n=600)
    w = GTAA(lookback=120, top_k=1, vol_window=40).generate_weights(data)
    tail = w.iloc[200:]
    cn = tail[realdata.ASSET_CLASSES["cn_equity"]].sum(axis=1).mean()
    assert cn > 0.5            # 强势的 A 股类别被重仓


def test_sixty_forty_constant_split():
    data = _mk(trends={c: 0.05 for c in realdata.ASSET_CLASSES}, vols={c: 0.15 for c in realdata.ASSET_CLASSES})
    w = SixtyForty(equity=0.6).generate_weights(data)
    eq = (realdata.ASSET_CLASSES["cn_equity"] + realdata.ASSET_CLASSES["global_equity"]
          + realdata.ASSET_CLASSES.get("cn_value", []))
    bond = realdata.ASSET_CLASSES["bond"]
    np.testing.assert_allclose(w[eq].sum(axis=1).values, 0.6, atol=1e-9)
    np.testing.assert_allclose(w[bond].sum(axis=1).values, 0.4, atol=1e-9)
    np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-9)


def test_deterministic_and_shape():
    data = _mk(trends={c: 0.08 for c in realdata.ASSET_CLASSES}, vols={c: 0.15 for c in realdata.ASSET_CLASSES})
    for cls in (AllWeather, GTAA, SixtyForty):
        a = cls().generate_weights(data)
        b = cls().generate_weights(data)
        pd.testing.assert_frame_equal(a, b)
        assert a.shape == (len(data.dates), len(data.symbols))
        assert list(a.columns) == data.symbols
