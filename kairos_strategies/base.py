"""策略与市场数据的核心契约。

所有策略统一实现 `generate_weights(data) -> 权重面板`：
index=日期、columns=资产、值=目标权重（引擎会自动滞后一期，杜绝未来函数）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """回测所需的市场数据容器（价格为主，成交量可选）。"""

    prices: pd.DataFrame
    volumes: Optional[pd.DataFrame] = None
    periods_per_year: int = 252
    name: str = "synthetic"

    @property
    def dates(self) -> pd.Index:
        return self.prices.index

    @property
    def symbols(self) -> List[str]:
        return list(self.prices.columns)

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    def returns(self, periods: int = 1) -> pd.DataFrame:
        """简单收益率面板（pct_change）。"""
        return self.prices.pct_change(periods).fillna(0.0)

    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices / self.prices.shift(1)).fillna(0.0)

    def rolling_vol(self, window: int = 20) -> pd.DataFrame:
        """滚动年化波动率。"""
        return self.returns().rolling(window).std() * np.sqrt(self.periods_per_year)


class Strategy(ABC):
    """策略基类。子类填充元信息并实现 generate_weights。

    元信息字段用于自动生成「研究记录」(QR 流程)：
      name        唯一标识（snake_case）
      channel     来源途径/类别（如 technical / momentum / meanrev ...）
      universe    'timing'(逐资产择时) 或 'cross_section'(截面选资产)，仅描述用
      long_only   是否只做多
      description 一句话思路
      hypothesis  核心假设（为什么可能有效）
      source      策略收集途径/灵感来源说明
      params      参数（dict）
    """

    name: str = "base"
    channel: str = "unknown"
    universe: str = "timing"
    long_only: bool = True
    description: str = ""
    hypothesis: str = ""
    source: str = ""
    params: Dict = {}

    @abstractmethod
    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        """返回目标权重面板 (index=data.dates, columns=data.symbols)。

        约定：
        - 只用「截至当期」的信息（引擎还会再滞后一期，双重保险）。
        - 权重可含 NaN（视为 0），引擎会对齐与裁剪。
        - long_only 策略权重应 >= 0；截面策略建议每行绝对值和 <= 1。
        """
        raise NotImplementedError

    def meta(self) -> Dict:
        return {
            "name": self.name,
            "channel": self.channel,
            "universe": self.universe,
            "long_only": self.long_only,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "source": self.source,
            "params": dict(self.params),
        }


def align_weights(w: pd.DataFrame, data: MarketData, long_only: Optional[bool] = None,
                  clip: float = 1.0) -> pd.DataFrame:
    """把任意权重输出对齐到 data 的 index/columns，填 NaN、裁剪到 [-clip, clip]。"""
    out = (w.reindex(index=data.dates, columns=data.symbols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0))
    if long_only:
        out = out.clip(lower=0.0, upper=clip)
    else:
        out = out.clip(lower=-clip, upper=clip)
    return out
