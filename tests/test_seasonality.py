"""seasonality 渠道测试。

覆盖：面板形状/列对齐/有限值、long_only>=0、每行和<=1+eps、确定性、元信息，
以及行为断言：
- turn_of_month：持仓只出现在月末最后 1 个 / 月初前 3 个交易日（手工核对日历）；
- weekday_effect / month_of_year：预热期空仓 + 只在「历史均值为正」的星期几/月份持有；
- 防未来函数：篡改 t 之后的价格/日期、或直接截断数据，t 及之前的权重必须逐位不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import MarketData, make_synthetic_universe
from kairos_strategies.channels.seasonality import (
    MonthOfYearStrategy,
    TurnOfMonthStrategy,
    WeekdayEffectStrategy,
)

ALL_CLASSES = (TurnOfMonthStrategy, WeekdayEffectStrategy, MonthOfYearStrategy)
LEARNED_CLASSES = (WeekdayEffectStrategy, MonthOfYearStrategy)


@pytest.fixture(scope="module")
def synth() -> MarketData:
    return make_synthetic_universe(n_assets=4, n_days=600, seed=7)


def _md(prices: pd.DataFrame) -> MarketData:
    return MarketData(prices=prices)


def _check_common(w: pd.DataFrame, data: MarketData) -> None:
    assert isinstance(w, pd.DataFrame)
    assert w.shape == (len(data.dates), len(data.symbols))
    assert list(w.columns) == list(data.symbols)
    assert w.index.equals(data.dates)
    v = w.values
    assert np.isfinite(v).all()
    assert (v >= 0.0).all()                              # long_only
    assert v.sum(axis=1).max() <= 1.0 + 1e-9             # 每行和 <= 1 + eps


def _monday_effect_data(n: int = 120) -> MarketData:
    """构造已知数据：仅周一收益为正（对数收益 +1%），其余星期几收益精确为 0。"""
    idx = pd.bdate_range("2021-01-04", periods=n)        # 周一起始
    r = np.where(idx.weekday == 0, 0.01, 0.0)
    r[0] = 0.0
    pa = 100.0 * np.exp(np.cumsum(r))
    pb = 50.0 * np.exp(np.cumsum(r * 0.5))
    return _md(pd.DataFrame({"A": pa, "B": pb}, index=idx))


def _january_effect_data(n: int = 1000) -> MarketData:
    """构造已知数据：仅一月的日收益为正（对数收益 +0.5%），其余月份精确为 0。"""
    idx = pd.bdate_range("2018-01-01", periods=n)        # 覆盖 2018~2021
    r = np.where(idx.month == 1, 0.005, 0.0)
    r[0] = 0.0
    pa = 100.0 * np.exp(np.cumsum(r))
    pb = 60.0 * np.exp(np.cumsum(r * 0.8))
    return _md(pd.DataFrame({"A": pa, "B": pb}, index=idx))


# ---------- 通用约束 ----------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_panel_shape_and_constraints(cls, synth):
    w = cls().generate_weights(synth)
    _check_common(w, synth)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_row_sum_zero_or_full_budget(cls, synth):
    w = cls().generate_weights(synth)
    rs = w.values.sum(axis=1)
    assert np.all((rs == 0.0) | np.isclose(rs, 1.0))     # 等权满仓或空仓，无杠杆


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_deterministic(cls, synth):
    w1 = cls().generate_weights(synth)
    w2 = cls().generate_weights(synth)
    pd.testing.assert_frame_equal(w1, w2)


def test_meta_contract():
    names = set()
    for cls in ALL_CLASSES:
        s = cls()                                        # 必须可无参构造
        assert s.channel == "seasonality"
        assert s.long_only is True
        assert s.universe in ("timing", "cross_section")
        assert s.name and s.description and s.hypothesis and s.source
        assert isinstance(s.params, dict)
        names.add(s.name)
    assert names == {"turn_of_month", "weekday_effect", "month_of_year"}


# ---------- 行为断言：turn_of_month ----------

def test_turn_of_month_holds_only_in_window():
    idx = pd.bdate_range("2021-01-01", periods=70)       # 2021-01-01 为周五
    prices = pd.DataFrame({"A": np.linspace(100, 120, 70),
                           "B": np.linspace(50, 40, 70)}, index=idx)
    data = _md(prices)
    w = TurnOfMonthStrategy().generate_weights(data)
    _check_common(w, data)
    # 手工核对日历：每月最后 1 个工作日 + 最前 3 个工作日
    hold = ["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-29",
            "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-26",
            "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-31",
            "2021-04-01"]
    flat = ["2021-01-06", "2021-01-15", "2021-01-28",
            "2021-02-04", "2021-02-17", "2021-02-25",
            "2021-03-04", "2021-03-30", "2021-04-06"]
    for d in hold:
        assert np.allclose(w.loc[d].values, 0.5), d      # 窗口内等权 1/2，行和=1
    for d in flat:
        assert w.loc[d].sum() == 0.0, d                  # 非窗口日空仓


# ---------- 行为断言：weekday_effect ----------

def test_weekday_effect_warmup_then_only_positive_weekday():
    data = _monday_effect_data(n=120)
    idx = data.dates
    w = WeekdayEffectStrategy().generate_weights(data)
    _check_common(w, data)
    min_obs = int(WeekdayEffectStrategy.params["min_obs"])   # 8
    warm = 5 * min_obs                                       # 第 40 行前周一历史样本 < min_obs
    assert w.values[:warm].sum() == 0.0                      # 预热期空仓
    rowsum = w.values[warm:].sum(axis=1)
    is_mon = idx.weekday.values[warm:] == 0
    assert np.allclose(rowsum[is_mon], 1.0)                  # 周一历史均值>0 -> 等权持有
    assert (rowsum[~is_mon] == 0.0).all()                    # 其余星期历史均值=0 -> 空仓


# ---------- 行为断言：month_of_year ----------

def test_month_of_year_warmup_then_only_positive_month():
    data = _january_effect_data(n=1000)
    idx = data.dates
    w = MonthOfYearStrategy().generate_weights(data)
    _check_common(w, data)
    # 预热期空仓：2018 全年任何月份的同月历史观测都不足 min_obs=40 -> 全 0
    assert w.loc[np.asarray(idx < pd.Timestamp("2019-01-01"))].values.sum() == 0.0
    # 历史充足后（2020 起）：仅一月（历史均值>0）等权持有，非一月（均值=0）空仓
    later = np.asarray(idx >= pd.Timestamp("2020-01-01"))
    rowsum = w.values[later].sum(axis=1)
    is_jan = idx.month.values[later] == 1
    assert np.allclose(rowsum[is_jan], 1.0)
    assert (rowsum[~is_jan] == 0.0).all()


# ---------- 防未来函数 ----------

@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_lookahead_tail_price_change(cls, synth):
    """篡改 t0 之后的全部价格，t0 及之前的权重必须逐位不变。"""
    s = cls()
    w_full = s.generate_weights(synth)
    t0 = 300
    p2 = synth.prices.copy()
    p2.iloc[t0 + 1:] = p2.iloc[t0 + 1:] * 3.7 + 11.0
    w2 = s.generate_weights(_md(p2))
    pd.testing.assert_frame_equal(w_full.iloc[: t0 + 1], w2.iloc[: t0 + 1])


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_lookahead_truncation(cls, synth):
    """只用前 k 行数据跑出的权重，必须等于全量权重的前 k 行（严格因果）。"""
    s = cls()
    w_full = s.generate_weights(synth)
    k = 250
    w_trunc = s.generate_weights(_md(synth.prices.iloc[:k]))
    pd.testing.assert_frame_equal(w_full.iloc[:k], w_trunc)


@pytest.mark.parametrize("cls,t0", [(WeekdayEffectStrategy, 70),
                                    (MonthOfYearStrategy, 600)])
def test_no_lookahead_tail_date_change(cls, t0):
    """改动 t0 之后的日期标签（整体平移 9 天，改变其星期/月份归属），
    t0 及之前的权重必须逐位不变。"""
    data = _monday_effect_data() if cls is WeekdayEffectStrategy else _january_effect_data()
    s = cls()
    w_full = s.generate_weights(data)
    idx = data.dates
    moved = pd.DatetimeIndex(np.where(np.arange(len(idx)) <= t0,
                                      idx.values,
                                      (idx + pd.Timedelta(days=9)).values))
    p2 = data.prices.copy()
    p2.index = moved
    w2 = s.generate_weights(_md(p2))
    # 重建索引后 freq 属性丢失，仅属元数据差异，比较时忽略 freq
    pd.testing.assert_frame_equal(w_full.iloc[: t0 + 1], w2.iloc[: t0 + 1],
                                  check_freq=False)
