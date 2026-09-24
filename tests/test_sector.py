"""sector 行业中性/均衡的离线测试。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import sector

SMAP = {"A": "x", "B": "x", "C": "y", "D": "y"}


def _w(rows):
    idx = pd.bdate_range("2021-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["A", "B", "C", "D"])


def test_make_sector_map():
    m = sector.make_sector_map({"x": ["A", "B"], "y": ["C"]})
    assert m == {"A": "x", "B": "x", "C": "y"}


def test_equalize_balances_active_sectors_and_preserves_total():
    w = _w([[0.6, 0.2, 0.1, 0.0]])  # 行业x=0.8, y=0.1, 总=0.9
    out = sector.sector_equalize(w, SMAP)
    gx = out[["A", "B"]].sum(axis=1).iloc[0]
    gy = out[["C", "D"]].sum(axis=1).iloc[0]
    assert gx == pytest.approx(gy, rel=1e-9)          # 活跃行业均衡
    assert gx + gy == pytest.approx(0.9, rel=1e-9)    # 总敞口不变
    assert (out.values >= -1e-12).all()               # 仍只做多


def test_equalize_zero_row_stays_zero():
    w = _w([[0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.4, 0.0]])
    out = sector.sector_equalize(w, SMAP)
    assert out.iloc[0].abs().sum() == 0.0
    # 第二行两行业各 0.4，已均衡，应基本不变
    assert out.iloc[1].sum() == pytest.approx(0.8, rel=1e-9)


def test_equalize_does_not_create_positions_in_unheld_sectors():
    w = _w([[0.5, 0.5, 0.0, 0.0]])  # 只持有行业 x
    out = sector.sector_equalize(w, SMAP)
    assert out[["C", "D"]].values.sum() == 0.0        # 未持有行业 y 不被凭空建仓
    assert out[["A", "B"]].sum(axis=1).iloc[0] == pytest.approx(1.0, rel=1e-9)


def test_neutralize_sector_and_dollar_neutral():
    w = _w([[0.4, 0.2, 0.3, 0.1]])
    out = sector.sector_neutralize(w, SMAP)
    assert out[["A", "B"]].sum(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-9)  # 行业x净0
    assert out[["C", "D"]].sum(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-9)  # 行业y净0
    assert out.sum(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-9)             # 美元中性
    assert out.abs().sum(axis=1).iloc[0] <= 1.0 + 1e-9                         # 绝对和≤cap


def test_neutralize_respects_cap():
    w = _w([[0.9, 0.0, 0.0, 0.0]])
    out = sector.sector_neutralize(w, SMAP, cap=0.5)
    assert out.abs().sum(axis=1).iloc[0] <= 0.5 + 1e-9


def test_deterministic_and_shape():
    w = _w([[0.3, 0.1, 0.2, 0.2], [0.0, 0.5, 0.5, 0.0]])
    a = sector.sector_equalize(w, SMAP)
    b = sector.sector_equalize(w, SMAP)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == w.shape and list(a.columns) == list(w.columns)
