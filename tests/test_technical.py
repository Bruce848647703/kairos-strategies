import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData
from kairos_strategies.channels.technical import (
    SmaCrossStrategy, RsiReversionStrategy, MacdTrendStrategy,
    BollingerBreakoutStrategy, DonchianTurtleStrategy, _to_weights,
)


def _data(prices_dict, n=None):
    prices = pd.DataFrame(prices_dict)
    return MarketData(prices=prices)


def test_sma_cross_long_in_uptrend_flat_in_downtrend():
    up = pd.DataFrame({"A": np.linspace(100, 200, 60)},
                      index=pd.bdate_range("2021-01-01", periods=60))
    down = pd.DataFrame({"A": np.linspace(200, 100, 60)},
                        index=pd.bdate_range("2021-01-01", periods=60))
    s = SmaCrossStrategy()
    w_up = s.generate_weights(_data({"A": up["A"]}))
    w_dn = s.generate_weights(_data({"A": down["A"]}))
    assert w_up["A"].iloc[-1] > 0        # 上升趋势末期持有
    assert w_dn["A"].iloc[-1] == 0.0     # 下降趋势空仓


def test_weights_equal_budget_cap():
    data = _data({"A": np.linspace(100, 110, 40), "B": np.linspace(100, 90, 40)})
    data.prices.index = pd.bdate_range("2021-01-01", periods=40)
    for cls in (SmaCrossStrategy, RsiReversionStrategy, MacdTrendStrategy,
                BollingerBreakoutStrategy, DonchianTurtleStrategy):
        w = cls().generate_weights(data)
        assert (w.values >= 0).all()
        assert w.values.sum(axis=1).max() <= 1.0 + 1e-9


def test_rsi_reversion_only_long_when_oversold():
    # 构造先跌后涨：下跌段 RSI 低 -> 应有持仓；强涨段 RSI 高 -> 空仓
    px = np.concatenate([np.linspace(100, 70, 40), np.linspace(70, 130, 40)])
    data = _data({"A": px})
    data.prices.index = pd.bdate_range("2021-01-01", periods=80)
    w = RsiReversionStrategy().generate_weights(data)
    assert w["A"].iloc[:40].max() > 0       # 下跌段出现过持仓
    assert w["A"].iloc[-5:].sum() == 0.0    # 末期强涨空仓


def test_donchian_breakout_enters_on_new_high():
    flat = np.full(40, 100.0)
    breakout = np.concatenate([flat, np.linspace(101, 130, 20)])
    data = _data({"A": breakout})
    data.prices.index = pd.bdate_range("2021-01-01", periods=60)
    w = DonchianTurtleStrategy().generate_weights(data)
    assert w["A"].iloc[45] > 0              # 突破后持有


def test_to_weights_scales_by_n_assets():
    data = _data({"A": np.ones(5), "B": np.ones(5), "C": np.ones(5), "D": np.ones(5)})
    data.prices.index = pd.bdate_range("2021-01-01", periods=5)
    sig = pd.DataFrame(1.0, index=data.dates, columns=data.symbols)
    w = _to_weights(sig, data)
    assert np.allclose(w.values, 0.25)
