"""realdata 的离线测试（不联网）：只测本地 CSV 加载与 universe 合法性。"""
import numpy as np
import pandas as pd
import pytest

from kairos_strategies import realdata


def _write(tmp_path, sym, closes, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    pd.DataFrame({
        "date": idx, "open": closes, "high": np.array(closes) * 1.01,
        "low": np.array(closes) * 0.99, "close": closes, "volume": 1000.0,
    }).to_csv(tmp_path / f"{sym}.csv", index=False)


def test_load_panel_basic(tmp_path):
    for sym, base in (("sh600000", 10.0), ("sz000001", 20.0)):
        _write(tmp_path, sym, np.linspace(base, base + 5, 8))
    md = realdata.load_panel(str(tmp_path))
    assert md.prices.shape == (8, 2)
    assert list(md.prices.columns) == ["sh600000", "sz000001"]
    assert md.name == "ashare_real"
    assert not md.prices.isna().any().any()
    assert md.volumes.shape == md.prices.shape


def test_load_panel_handles_halt_and_listing(tmp_path):
    # A 全程上市；B 前 3 天未上市(NaN) + 中间一天停牌(0 价)
    idx = pd.bdate_range("2020-01-01", periods=8)
    a = pd.DataFrame({"date": idx, "open": 10, "high": 11, "low": 9,
                      "close": np.arange(8) + 10.0, "volume": 100})
    bc = [np.nan, np.nan, np.nan, 20.0, 0.0, 21.0, 22.0, 23.0]  # 0 价=停牌
    b = pd.DataFrame({"date": idx, "open": 20, "high": 21, "low": 19,
                      "close": bc, "volume": 100})
    a.to_csv(tmp_path / "sh600000.csv", index=False)
    b.to_csv(tmp_path / "sz000001.csv", index=False)
    md = realdata.load_panel(str(tmp_path), drop_incomplete=True)
    assert not md.prices.isna().any().any()      # 无 NaN
    assert (md.prices.values > 0).all()          # 停牌 0 价被前填为正
    assert len(md.prices) == 5                    # 裁到全体上市之后(8-3)


def test_load_panel_missing_dir_raises(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        realdata.load_panel(str(empty))


def test_universe_wellformed():
    syms = list(realdata.UNIVERSE)
    assert len(syms) == len(set(syms))
    for s in syms:
        assert s[:2] in ("sh", "sz") and s[2:].isdigit() and len(s[2:]) == 6
