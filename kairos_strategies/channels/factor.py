"""渠道：factor —— 因子/异象截面策略（cross_section，只做多）。

收集途径：学术与实务中的经典「价格/波动」异象（low-volatility、short-term
reversal、trend quality/price efficiency、residual momentum）。全部为本仓库
原创实现，**仅用价格与波动构造**，不引入任何基本面数据。

统一范式：对每个交易日，用「截至当期」的信息计算一个截面因子分数（越高越
好），做多分数排名前 ``top_frac`` 的一篮子；选中集内按分数线性加权（分数越高
权重越大），逐行归一到和≈1。预热期（信息不足）整行权重为 0。所有滚动/差分/
排名均只看历史，杜绝未来函数（引擎还会再滞后一期，双重保险）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..base import MarketData, Strategy


def _top_fraction_weights(scores: pd.DataFrame, top_frac: float,
                          eps: float = 1e-6) -> pd.DataFrame:
    """把「越高越好」的因子分数转成只做多、截面归一的目标权重面板。

    步骤（逐行/逐日，仅用当期截面信息）：
      1. 分数降序排名，选中前 ``k = ceil(top_frac * N)`` 名做多，其余为 0；
         NaN 分数（预热期）一律不选。
      2. 以选中集内的最低分（即第 k 名分数）为基准 ``bnd``，令
         ``raw = score - bnd + eps``：分数越高 raw 越大，且 raw 恒 > 0。
      3. 逐行归一 ``w = raw / Σraw``，使选中行权重和≈1；无选中（预热）行为 0。

    这样既保证「因子越极端权重越高」的单调倾斜，又对离群值稳健（线性而非指数）。
    """
    scores = scores.replace([np.inf, -np.inf], np.nan)
    n = scores.shape[1]
    k = max(1, int(np.ceil(top_frac * n - 1e-9)))
    # 降序排名：分数最高者 rank=1；NaN 保持 NaN（不会被选中）
    ranks = scores.rank(axis=1, ascending=False, method="min", na_option="keep")
    sel = (ranks <= k) & scores.notna()
    # 选中集内的最低分 = 第 k 名分数，作为线性倾斜的基准
    bnd = scores.where(sel).min(axis=1)
    raw = (scores.sub(bnd, axis=0) + eps).where(sel, 0.0).clip(lower=0.0)
    row_sum = raw.sum(axis=1)
    w = raw.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)
    return w.clip(lower=0.0, upper=1.0)


class LowVolatilityStrategy(Strategy):
    name = "low_volatility"
    channel = "factor"
    universe = "cross_section"
    long_only = True
    description = "低波动异象：按滚动已实现波动升序排名，做多波动最低的一篮子，波动越低权重越高。"
    hypothesis = ("高波动『彩票型』资产因投资者偏好博取高收益而被系统性高估，加之机构受杠杆约束"
                  "倾向用高 beta 替代加杠杆，使低波动组合长期风险调整后收益更优。当低波资产拥挤/"
                  "估值过高，或危机中相关性趋同、市场普跌时，该优势可能减弱甚至反转。")
    source = "因子异象——低波动 (low-volatility anomaly)，纯价格/波动构造。"
    params = {"vol_window": 21, "top_frac": 1 / 3}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        vol = ind.realized_vol(data.prices, self.params["vol_window"],
                               data.periods_per_year)
        score = -vol  # 波动越低，分数越高
        return _top_fraction_weights(score, self.params["top_frac"])


class ShortTermReversalStrategy(Strategy):
    name = "short_term_reversal"
    channel = "factor"
    universe = "cross_section"
    long_only = True
    description = "短期反转：按过去短窗口收益升序排名，做多近期跌幅最大的一篮子，跌得越多权重越高。"
    hypothesis = ("流动性冲击与投资者过度反应使近端价格短暂偏离，随后做市商回补与均值回归推动反弹；"
                  "做多最近大幅下跌的一篮子可捕捉这种短期修复。当下跌由基本面恶化（信息驱动）主导，"
                  "或市场处于持续单边趋势时，反转会失效甚至放大亏损。")
    source = "因子异象——短期反转 (short-term reversal)，纯价格构造。"
    params = {"lookback": 5, "top_frac": 1 / 3}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        mom = ind.momentum(data.prices, self.params["lookback"])
        score = -mom  # 近期收益越负（跌得越多），分数越高
        return _top_fraction_weights(score, self.params["top_frac"])


class TrendQualityStrategy(Strategy):
    name = "trend_quality"
    channel = "factor"
    universe = "cross_section"
    long_only = True
    description = "趋势质量/价格效率：用净位移/路径长度（效率比）衡量趋势平滑度，做多效率最高的一篮子。"
    hypothesis = ("净位移占路径长度比例高，说明趋势由持续信息驱动而非噪声来回拉锯，动量更可信、"
                  "回撤更小；做多『走得直』的资产可过滤假突破与震荡。在趋势末端发生反转，或市场进入"
                  "高噪声状态时，效率比会失去 predictive 能力。")
    source = "因子异象——趋势质量/价格效率 (efficiency ratio)，纯价格构造。"
    params = {"window": 30, "top_frac": 1 / 3}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = data.prices
        n = self.params["window"]
        disp = (p - p.shift(n)).abs()                       # 净位移
        path = p.diff().abs().rolling(n, min_periods=n).sum()  # 路径长度
        eff = disp / path.replace(0.0, np.nan)              # 效率比 ∈ [0,1]
        return _top_fraction_weights(eff, self.params["top_frac"])


class IdioMomentumStrategy(Strategy):
    name = "idio_momentum"
    channel = "factor"
    universe = "cross_section"
    long_only = True
    description = "残差动量：用过去收益减去等权市场收益得到相对强度，做多相对强度靠前的一篮子。"
    hypothesis = ("剔除等权市场收益后的相对强度更接近资产特质信息的扩散速度，比原始动量更少被市场"
                  "beta 污染，截面区分度更高、更稳健。当市场剧烈轮动、特质收益被系统性因子主导，或"
                  "残差动量策略拥挤时，其优势会衰减。")
    source = "因子异象——残差/相对动量 (residual momentum)，纯价格构造。"
    params = {"window": 60, "top_frac": 1 / 3}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        n = self.params["window"]
        mom = ind.momentum(data.prices, n)
        mkt = mom.mean(axis=1)            # 等权市场收益（截面均值，仅用当期信息）
        score = mom.sub(mkt, axis=0)      # 残差/相对动量，越高越好
        return _top_fraction_weights(score, self.params["top_frac"])
