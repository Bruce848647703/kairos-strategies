"""渠道：long_short —— 多空股票 alpha（截面因子排序选股，美元中性，逐行权重和 ≈ 0）。

收集途径与范式（**全部为本仓库原创实现**，只用 numpy/pandas；因子与排序辅助函数
均在本模块内自实现，不 import 其它渠道的私有函数）：

统一构造：对每个交易日，用「截至当期」的信息给每个资产打一个截面分数（越高越
值得做多），按分数排序**做多 top 分位一篮子、做空 bottom 分位一篮子**，两腿各自
等权归一到 ``leg_budget``（默认 0.5）预算——逐行权重和恒 ≈ 0（美元/市场中性），
绝对值之和 = 2 * leg_budget ≤ 1（不加杠杆）。有效分数不足 2 个（凑不出多空两腿）
或处于预热期（因子信息不足）的行整行空仓。

与相邻渠道的区别（避免重复）：
  * factor 渠道的排序策略**只做多**（每行和 ≈ 1）；本渠道同时做空 bottom 分位，
    是它们的多空/市场中性版本，收益来源是截面价差而非方向暴露；
  * statarb 渠道（已做美元中性）交易的是**统计结构偏离**（协整价差、PCA 公允
    价格、篮子价差 z-score）；本渠道是经典「按 alpha 分数截面排序的多空选股」，
    信号来自因子分位本身，不涉及价差收敛假设；
  * momentum 渠道的 xs_momentum 只做多 top 1/3；本渠道 xs_momentum_ls 两端下注。

防未来函数：所有因子只用滚动**尾部窗口** / shift（历史）价格与当期截面信息计算；
引擎还会再滞后一期，双重保险。排序并列按列索引稳定打破（lexsort），同一数据多次
调用结果逐位一致（确定性），且截断样本重算得到的前缀权重与全样本一致（前缀不变）。

五个策略：
  1. ``xs_momentum_ls``        截面动量(12-1)多空
  2. ``reversal_ls``           短期反转多空
  3. ``quality_ls``            质量（价格效率 + 低波动）合成多空
  4. ``residual_momentum_ls``  残差动量（回归掉等权市场收益）多空
  5. ``lowvol_ls``             低波动异象多空
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import MarketData, Strategy

_EPS = 1e-9      # 排名/取整容差
_TINY = 1e-12    # 除零 / 退化保护阈值


# ------------------------------------------------------------------ 通用工具（本模块自实现）

def _prices(data: MarketData) -> pd.DataFrame:
    """价格面板 -> float64、裁到正数（保证比值/对数安全），对齐 index/columns。"""
    return data.prices.astype("float64").clip(lower=1e-9)


def _returns(data: MarketData) -> pd.DataFrame:
    """日度简单收益面板；首期无昨收，置 0（非未来信息），其余 NaN 一律填 0。"""
    p = _prices(data)
    r = p.pct_change()
    if len(r) > 0:
        r.iloc[0] = 0.0
    return r.fillna(0.0)


def _realized_vol(r: pd.DataFrame, window: int, periods_per_year: int) -> pd.DataFrame:
    """尾部窗口年化已实现波动率（只用历史，min_periods=window，预热期为 NaN）。"""
    w = int(window)
    return r.rolling(w, min_periods=w).std() * np.sqrt(float(periods_per_year))


def _row_z(x: pd.DataFrame) -> pd.DataFrame:
    """截面标准化：逐行减截面均值、除截面标准差（ddof=1，忽略 NaN）。

    只用**当期截面**信息，不涉及任何未来数据；某行有效值不足 2 个或标准差 ≈ 0
    （整行退化）时返回 NaN —— 该资产当日不参与排序。NaN 原样保留（预热期不选）。
    """
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > _TINY), axis=0)


def _quantile_long_short(scores: pd.DataFrame, data: MarketData, top_frac: float,
                         leg_budget: float = 0.5) -> pd.DataFrame:
    """把「越高越好」的截面分数转成美元中性、无杠杆的多空目标权重面板。

    逐行（逐日，只用当期截面分数）：
      1. 有效（非 NaN）分数降序排序，前 ``k`` 名做多、后 ``k`` 名做空，
         ``k = clip(ceil(top_frac * n_valid), 1, n_valid // 2)``，保证两腿非空且
         互不重叠；并列分数按「列索引升序」稳定打破（``np.lexsort``），结果确定；
      2. 两腿内部等权：多头 +leg_budget/k、空头 -leg_budget/k，两腿各自归一到
         ``leg_budget``（默认 0.5）预算 —— 每行权重和 ≈ 0（多空预算精确抵消），
         绝对值之和 = 2 * leg_budget ≤ 1（不加杠杆）；
      3. 有效分数少于 2 个（凑不出多空两腿）-> 整行空仓（全 0）。
    输出对齐 ``data`` 的 index/columns，不含 NaN。
    """
    aligned = (scores.reindex(index=data.dates, columns=data.symbols)
                     .apply(pd.to_numeric, errors="coerce"))
    v = aligned.to_numpy(dtype="float64")
    v[~np.isfinite(v)] = np.nan
    T, n = v.shape
    out = np.zeros((T, n), dtype="float64")
    frac = float(top_frac)
    budget = float(leg_budget)
    for t in range(T):
        row = v[t]
        idx = np.flatnonzero(np.isfinite(row))
        nv = int(idx.size)
        if nv < 2:
            continue                                          # 凑不出多空两腿 -> 空仓
        k = int(np.ceil(nv * frac - _EPS))                    # top/bottom 分位的名额
        k = max(1, min(k, nv // 2))                           # 两腿非空且不重叠
        vals = row[idx]
        order = np.lexsort((idx, -vals))                      # 主键分数降序，并列按列索引
        out[t, idx[order[:k]]] = budget / k                   # 多头腿：等权 +0.5/k
        out[t, idx[order[nv - k:]]] = -budget / k             # 空头腿：等权 -0.5/k
    return pd.DataFrame(out, index=data.dates, columns=data.symbols)


# ------------------------------------------------------------------ 策略 1：截面动量多空

class CrossSectionMomentumLongShortStrategy(Strategy):
    name = "xs_momentum_ls"
    channel = "long_short"
    universe = "cross_section"
    long_only = False
    description = ("截面动量多空(12-1)：按 t-lookback 至 t-skip 的累计收益给全 universe 截面排序，"
                   "做多最强 top 分位一篮子、做空最弱 bottom 分位一篮子，两腿等权各占 0.5 预算，"
                   "组合逐行权重和恒 ≈ 0（美元中性、毛敞口 = 1）。")
    hypothesis = ("成因：信息扩散有滞后，投资者对特质信息先反应不足（强势延续）、对代表性证据又"
                  "过度外推，使 3~12 个月的相对强弱有延续性；同时做多强者、做空弱者能把「强势延续」"
                  "与「弱势续差」两段价差一起赚到手，且美元中性构造剥离了市场 beta，收益主要来自"
                  "截面离散度而非大盘方向。失效条件：动量崩溃——暴跌后的 V 型反弹里前期最弱篮子"
                  "弹性最大，空头腿与多头腿同时受损；风格急切换或流动性冲击使截面相关性骤升、"
                  "离散度消失；排序被单一事件驱动的价格跳变主导，或策略拥挤后 alpha 被提前交易掉。")
    source = ("多空股票 alpha 经典范式——横截面动量 (Jegadeesh-Titman 12-1 风格) 的市场中性版本，"
              "本仓库原创实现；与 momentum 渠道的 xs_momentum（只做多 top 1/3）不同，此处补上"
              "做空 bottom 分位的另一条腿，且排序/配权工具函数均在本模块内自实现。")
    params = {"lookback": 252,          # 动量观察长窗口（约 12 个月）
              "skip": 21,               # 剔除最近约 1 个月，规避短期反转污染
              "top_frac": 1.0 / 3.0,    # 多空两腿各取的分位（1/3 分位）
              "leg_budget": 0.5}        # 单腿预算：多头 +0.5、空头 -0.5，行和 ≈ 0

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _prices(data)
        lb = int(self.params["lookback"])
        sk = int(self.params["skip"])
        # t-lb 至 t-sk 的累计收益：只用 shift（历史）价格，绝不偷看未来
        score = p.shift(sk) / p.shift(lb) - 1.0
        return _quantile_long_short(score, data, self.params["top_frac"],
                                    self.params["leg_budget"])


# ------------------------------------------------------------------ 策略 2：短期反转多空

class ReversalLongShortStrategy(Strategy):
    name = "reversal_ls"
    channel = "long_short"
    universe = "cross_section"
    long_only = False
    description = ("短期反转多空：按最近 lookback 日收益截面排序，做多近期跌得最多的 bottom 分位、"
                   "做空近期涨得最多的 top 分位（分数 = -近端收益），两腿等权各占 0.5 预算，"
                   "组合美元中性。")
    hypothesis = ("成因：短窗口（周级）价格变动里流动性冲击与过度反应成分大——被动抛压、止损盘与"
                  "情绪化追涨把价格暂时推离短期均衡，做市商回补与套利资金随后推动修复：近端超跌者"
                  "反弹、超涨者回吐，多空两端同时赚取这个截面收敛价差，市场方向被对冲掉。失效条件："
                  "短窗口涨跌由真实信息（业绩爆雷/超预期、并购）驱动时，跌者继续跌、涨者继续涨，"
                  "反转变成「接飞刀」与「空逼空」；持续单边资金流/动量行情中信号长期为负；信号窗口"
                  "短、换手高，交易成本与滑点可能吞掉毛 alpha；空头腿对被逼空的高波动标的风险不封顶。")
    source = ("多空股票 alpha 经典范式——短期反转 (short-term reversal) 的市场中性版本，本仓库原创"
              "实现；与 factor 渠道的 short_term_reversal（只做多超跌篮子）、statarb 渠道的 "
              "xs_zscore_reversion（对全篮子连续 z 配权）不同，此处是「top/bottom 分位各取一篮子、"
              "腿内等权」的分位多空形态。")
    params = {"lookback": 5,            # 近端收益观察窗口（约一周）
              "top_frac": 1.0 / 3.0,    # 多空两腿各取的分位
              "leg_budget": 0.5}        # 单腿预算

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _prices(data)
        lb = max(int(self.params["lookback"]), 1)
        mom = p / p.shift(lb) - 1.0                            # 只用历史价格
        score = -mom                                           # 近期跌得越多，分数越高
        return _quantile_long_short(score, data, self.params["top_frac"],
                                    self.params["leg_budget"])


# ------------------------------------------------------------------ 策略 3：质量合成多空

class QualityLongShortStrategy(Strategy):
    name = "quality_ls"
    channel = "long_short"
    universe = "cross_section"
    long_only = False
    description = ("质量多空：用「带符号价格效率（净位移/路径长度）+ 低波动」的截面标准化合成分"
                   "衡量趋势质量，做多走得又直又稳的 top 分位、做空震荡剧烈的 bottom 分位，"
                   "两腿等权各占 0.5 预算，组合美元中性。")
    hypothesis = ("成因：净位移占路径长度比例高，说明行情由持续的信息流驱动而非噪声来回拉锯，"
                  "这类「干净趋势」的动量更可信、回撤更小；低波动分量则捕捉彩票型高波资产被系统性"
                  "高估的异象。两个成分先做截面标准化再加权合成，量纲统一、互不淹没，选出的多头是"
                  "「趋势平滑且波动低」、空头是「震荡剧烈且波动高」的资产，价差来自信息驱动收益与"
                  "噪声驱动收益的分化。失效条件：趋势末端——效率比记录的是历史平滑度，拐点处高质量"
                  "资产率先反转；波动结构突变（低波标的遭遇黑天鹅）使低波分量反向；高噪声环境下两个"
                  "成分的截面区分度同时下降；合成权重是主观先验，成分间相关性结构变化会让合成分退化"
                  "为单一因子。")
    source = ("多空股票 alpha 范式——质量因子（价格效率 efficiency ratio + low-volatility 合成）的"
              "市场中性版本，本仓库原创实现，纯价格/波动构造、不臆造基本面；与 factor 渠道的 "
              "trend_quality / low_volatility（各自只做多、单一成分）不同，此处是双成分截面标准化"
              "合成后两端下注。")
    params = {"window": 60,             # 价格效率（净位移/路径长度）窗口
              "vol_window": 21,         # 已实现波动窗口（约一个月）
              "eff_weight": 0.5,        # 合成分中价格效率的权重
              "vol_weight": 0.5,        # 合成分中低波动的权重
              "top_frac": 1.0 / 3.0,    # 多空两腿各取的分位
              "leg_budget": 0.5}        # 单腿预算

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        p = _prices(data)
        w = int(self.params["window"])
        disp = p - p.shift(w)                                    # 净位移（带符号）
        path = p.diff().abs().rolling(w, min_periods=w).sum()    # 路径长度
        eff = disp / path.replace(0.0, np.nan)                   # 带符号价格效率 ∈ [-1, 1]
        vol = _realized_vol(_returns(data), int(self.params["vol_window"]),
                            data.periods_per_year)
        # 两成分先截面标准化（当期截面信息），再按权重合成；波动越低分数越高
        score = (float(self.params["eff_weight"]) * _row_z(eff)
                 - float(self.params["vol_weight"]) * _row_z(vol))
        return _quantile_long_short(score, data, self.params["top_frac"],
                                    self.params["leg_budget"])


# ------------------------------------------------------------------ 策略 4：残差动量多空

class ResidualMomentumLongShortStrategy(Strategy):
    name = "residual_momentum_ls"
    channel = "long_short"
    universe = "cross_section"
    long_only = False
    description = ("残差动量多空：对每个资产用尾部窗口把它对「等权市场收益」做过原点回归得到 beta，"
                   "残差 = 自身收益 - beta×市场收益，按 score_window 期累计残差（以残差波动标准化）"
                   "截面排序，做多残差动量最强的 top 分位、做空最弱的 bottom 分位，两腿等权各占 "
                   "0.5 预算，组合美元中性。")
    hypothesis = ("成因：原始动量被市场 beta 污染——高 beta 资产在上涨期「看起来强」，其强势只是市场"
                  "的放大镜像，一旦市场回调便加倍回吐；把等权市场收益回归掉之后，累计残差更接近特质"
                  "信息（业绩、竞争力、资金关注度）的逐步扩散速度，截面区分度更高、时序更稳定，且多空"
                  "两端再对冲掉残余市场敞口，赚的是纯特质相对强弱。失效条件：等权市场收益是粗糙的单"
                  "因子代理，当真实因子结构多元（行业/风格主导）或 beta 不稳定时，残差仍被共同因子"
                  "污染，排序失真；市场急转弯处尾部窗口估出的 beta 滞后于真实敞口；特质收益由一次性"
                  "事件（跳空、复牌）贡献时残差动量追错方向；策略拥挤后残差动量价差被提前收敛。")
    source = ("多空股票 alpha 范式——残差动量 (residual momentum, Blitz-Huij-Martens 思路) 的市场"
              "中性版本，本仓库原创 numpy/pandas 实现（滚动过原点回归 beta + 波动标准化累计残差，"
              "不用 statsmodels）；与 factor 渠道的 idio_momentum（只做多、且用「动量减截面均值」"
              "近似而非回归残差）在信号构造与组合形态上均不同。")
    params = {"reg_window": 60,         # 滚动回归 beta 的尾部窗口
              "score_window": 60,       # 累计残差（残差动量）的观察窗口
              "top_frac": 1.0 / 3.0,    # 多空两腿各取的分位
              "leg_budget": 0.5}        # 单腿预算

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        r = _returns(data)
        mkt = r.mean(axis=1)                                     # 等权市场收益（当期截面）
        w = max(int(self.params["reg_window"]), 2)
        # 滚动过原点回归 beta_i = Σ(r_i·mkt) / Σ(mkt²)，只用尾部窗口，逐日更新
        num = r.mul(mkt, axis=0).rolling(w, min_periods=w).sum()
        den = (mkt * mkt).rolling(w, min_periods=w).sum()
        beta = num.div(den.where(den > _TINY), axis=0)
        resid = r - beta.mul(mkt, axis=0)                        # 回归掉市场后的残差收益
        h = max(int(self.params["score_window"]), 2)
        cum = resid.rolling(h, min_periods=h).sum()              # 累计残差动量
        scale = resid.rolling(h, min_periods=h).std() * np.sqrt(float(h))
        score = cum / scale.where(scale > _TINY)                 # 以残差波动为单位，截面可比
        return _quantile_long_short(score, data, self.params["top_frac"],
                                    self.params["leg_budget"])


# ------------------------------------------------------------------ 策略 5：低波动多空

class LowVolLongShortStrategy(Strategy):
    name = "lowvol_ls"
    channel = "long_short"
    universe = "cross_section"
    long_only = False
    description = ("低波动多空：按尾部窗口已实现波动率截面排序，做多波动最低的 top 分位一篮子、"
                   "做空波动最高的 bottom 分位一篮子（分数 = -波动率），两腿等权各占 0.5 预算，"
                   "组合美元中性，是低波异象的市场中性版本。")
    hypothesis = ("成因：彩票偏好使散户系统性高估「博收益」的高波动资产（愿意为其支付溢价），机构"
                  "又因杠杆/基准约束偏好高 beta 替代加杠杆，两头挤压之下高波资产预期收益反而更低；"
                  "同时做空高波、做多低波能把这个异象的两端价差都拿到手，且对冲掉市场方向后，剩余"
                  "暴露主要是「波动率溢价」这一维。失效条件：低波篮子拥挤、估值被买贵后异象衰减；"
                  "危机后修复期与流动性宽松期高 beta 资产暴力反弹，空头腿亏损可能远超多头腿盈利"
                  "（低波多空的最大回撤来源）；利率快速上行时低波资产（类债券久期属性）跑输；"
                  "低波篮子集中于防御性行业，行业轮动会造成阶段性风格回撤。")
    source = ("多空股票 alpha 经典范式——低波动异象 (low-volatility anomaly, Ang-Hodges 思路) 的"
              "市场中性版本，本仓库原创实现；与 factor 渠道的 low_volatility（只做多低波篮子）、"
              "statarb 渠道的 basket_neutral（用波动分篮后交易篮子便宜度价差）不同，此处直接按波动"
              "分位两端下注，波动率工具函数在本模块内自实现。")
    params = {"vol_window": 42,         # 已实现波动窗口（约两个月）
              "top_frac": 1.0 / 3.0,    # 多空两腿各取的分位
              "leg_budget": 0.5}        # 单腿预算

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        vol = _realized_vol(_returns(data), int(self.params["vol_window"]),
                            data.periods_per_year)
        score = -vol                                             # 波动越低，分数越高
        return _quantile_long_short(score, data, self.params["top_frac"],
                                    self.params["leg_budget"])
