"""渠道：seasonality —— 日历/季节性效应策略（等权组合择时，只做多）。

收集途径：日历效应文献范式（turn-of-the-month / day-of-the-week / month-of-the-year），
全部为本仓库原创实现，输出统一为目标权重面板。

工程要点（防未来函数）：
- turn_of_month：窗口判定只用「当日日期」的确定性日历算术（月内第几个工作日、
  加 N 个工作日是否跨月），不依赖任何未来价格，也不回看指数的未来行。
- weekday_effect / month_of_year：分星期/分月份的历史均值统计一律
  「组内 shift(1) + expanding 累计」，即第 t 行的判定只用严格早于 t 的同组收益，
  当期收益不参与当期判定；历史样本不足 min_obs 时视为预热期，一律空仓。

所有策略离线、确定性：同一 MarketData 多次调用结果一致；不使用成交量与外部数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy


def _equal_weight_panel(mask: np.ndarray, data: MarketData) -> pd.DataFrame:
    """把 (T,) 布尔持有掩码转成等预算权重面板：窗口内全资产各 1/N，窗口外 0。"""
    n = max(data.n_assets, 1)
    hold = np.asarray(mask, dtype=bool)[:, None]
    w = np.tile(np.where(hold, 1.0 / n, 0.0), (1, n))
    return pd.DataFrame(w, index=data.dates, columns=data.symbols)


def _expanding_group_mean_before(values: np.ndarray, codes: np.ndarray,
                                 min_obs: int) -> np.ndarray:
    """按分组（如星期几/月份）计算 expanding 历史均值，严格只用「当前行之前」的同组观测。

    防未来函数实现：先在组内 shift(1)（剔除当期观测），再做组内累计和 / 累计计数
    （两者均为因果运算），故第 t 行结果只取决于严格早于 t 的同组样本。
    同组历史样本数不足 min_obs 时返回 NaN（预热期，调用方应视为不持有）。
    """
    s = pd.Series(np.asarray(values, dtype=float))
    g = pd.Series(np.asarray(codes, dtype=np.int64))
    prev = s.groupby(g).shift(1)
    csum = prev.groupby(g).cumsum()
    ccnt = prev.notna().astype("int64").groupby(g).cumsum()
    mean = csum / ccnt.where(ccnt > 0)
    return mean.where(ccnt >= min_obs).values


def _portfolio_return(data: MarketData) -> np.ndarray:
    """等权组合日收益（横截面均值），作为日历效应统计的样本序列。"""
    return data.returns().mean(axis=1).values


class TurnOfMonthStrategy(Strategy):
    """月初/月末效应：只在每月最后 end_days 个与最前 start_days 个交易日持有等权组合。

    防未来函数：判定完全基于「当日日期」的日历算术——
    月初窗口 = 当月第几个工作日（np.busday_count，只数当月已过去的日子）；
    月末窗口 = 当日加 end_days 个工作日是否跨月（跨月 <=> 处于当月最后 end_days 个
    工作日）。二者都不读取未来价格，也不使用数据索引中 t 之后的行。
    """

    name = "turn_of_month"
    channel = "seasonality"
    universe = "timing"
    long_only = True
    description = "月初/月末效应：仅在每月最后 end_days 个与最前 start_days 个交易日持有等权组合，其余空仓。"
    hypothesis = ("真实市场常见假说：月末/月初有工资入账与定投资金持续流入、机构 window "
                  "dressing 与月度再平衡，使该窗口收益系统性偏高。注意：在合成随机数据上"
                  "通常无显著 edge，结果依赖数据本身是否含有该效应。")
    source = "日历效应文献范式——turn-of-the-month effect，本仓库原创实现。"
    params = {"end_days": 1, "start_days": 3}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        dates = pd.DatetimeIndex(data.prices.index)
        n_end = max(int(self.params["end_days"]), 1)
        n_start = max(int(self.params["start_days"]), 0)
        is_bday = dates.dayofweek.values < 5
        # 月初窗口：当月工作日序号 <= n_start（busday_count 只统计当月月初至当日之前）
        month_first = dates.to_period("M").to_timestamp()
        rank_in_month = np.busday_count(
            month_first.values.astype("datetime64[D]"),
            dates.values.astype("datetime64[D]"),
        ) + 1
        start_win = is_bday & (rank_in_month <= n_start)
        # 月末窗口：当日加 n_end 个工作日跨月 <=> 当日处于当月最后 n_end 个工作日内
        crosses_month = np.asarray((dates + pd.offsets.BDay(n_end)).month != dates.month)
        end_win = is_bday & crosses_month
        return _equal_weight_panel(start_win | end_win, data)


class WeekdayEffectStrategy(Strategy):
    """星期效应：只在「历史平均收益为正」的星期几持有等权组合。

    防未来函数：t 期判定所用的分星期历史均值，只由严格早于 t 的同星期几收益
    （组内 shift(1) 后 expanding 累计）构成，当期收益不参与当期判定；且某星期几
    需累计至少 min_obs 个历史观测才参与判定，预热期一律空仓。
    """

    name = "weekday_effect"
    channel = "seasonality"
    universe = "timing"
    long_only = True
    description = "星期效应：仅当当日星期几的历史（截至当期之前）平均收益为正时持有等权组合，否则空仓。"
    hypothesis = ("真实市场常见假说：周末消息积压与个人投资者情绪导致 Monday effect、"
                  "周中财报/公告节奏使某些星期几系统性偏强。注意：在合成随机数据上"
                  "通常无显著 edge，历史均值的符号近似随机，结果依赖数据本身。")
    source = "日历效应文献范式——day-of-the-week effect，本仓库原创实现。"
    params = {"min_obs": 8}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        dates = pd.DatetimeIndex(data.prices.index)
        hist_mean = _expanding_group_mean_before(
            _portfolio_return(data), dates.weekday.values, int(self.params["min_obs"]))
        mask = np.asarray(hist_mean) > 0.0   # NaN（预热期）比较为 False -> 空仓
        return _equal_weight_panel(mask, data)


class MonthOfYearStrategy(Strategy):
    """月份季节性：只在「历史平均收益为正」的日历月份持有等权组合。

    防未来函数：t 期判定所用的分月份历史均值，只由严格早于 t 的同月份日度收益
    （组内 shift(1) 后 expanding 累计）构成；同月份需累计至少 min_obs 个日度历史
    观测（约两个完整月份）才参与判定，预热期一律空仓。
    """

    name = "month_of_year"
    channel = "seasonality"
    universe = "timing"
    long_only = True
    description = "月份季节性：仅当当日所处月份的历史（截至当期之前）平均日收益为正时持有等权组合，否则空仓。"
    hypothesis = ("真实市场常见假说：一月效应（税收卖出回补、奖金流入）、Sell in May"
                  "（夏季成交清淡）、基金年初建仓与年末冲业绩等带来月份季节性。"
                  "注意：在合成随机数据上通常无显著 edge，结果依赖数据本身。")
    source = "日历效应文献范式——month-of-the-year seasonality，本仓库原创实现。"
    params = {"min_obs": 40}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        dates = pd.DatetimeIndex(data.prices.index)
        hist_mean = _expanding_group_mean_before(
            _portfolio_return(data), dates.month.values, int(self.params["min_obs"]))
        mask = np.asarray(hist_mean) > 0.0   # NaN（预热期）比较为 False -> 空仓
        return _equal_weight_panel(mask, data)
