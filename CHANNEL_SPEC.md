# CHANNEL_SPEC —— 新增策略渠道规范

你要为 `kairos-strategies` 增加**一个渠道模块**（若干原创策略）。严格遵循本规范，
风格对齐已完成的样板渠道 `kairos_strategies/channels/technical.py` 与 `benchmark.py`。

## 铁律
1. **100% 原创**，禁止复制任何第三方代码；可实现通用算法/范式。
2. **只用 numpy/pandas**；可 `from .. import indicators as ind` 复用自研指标。禁止联网、禁止真实交易。
3. **离线 + 确定性**：同一 `MarketData` 多次调用 `generate_weights` 结果必须一致。
4. **防未来函数**：第 t 期权重只能用截至 t 的信息（引擎还会再滞后一期，但你自己也不许偷看未来）。
5. Python 3.9 兼容，`from __future__ import annotations`，中文 docstring，类型注解。
6. **只创建你自己的两个文件**：`kairos_strategies/channels/<你的渠道>.py` 和 `tests/test_<你的渠道>.py`。
   不要改动 base/engine/registry/report/indicators/data、其它渠道、或任何已有文件。

## 你拿到的数据契约（只读）
```python
from ..base import MarketData, Strategy

data.prices   # pd.DataFrame, index=DatetimeIndex(约1000个工作日), columns=['A0'..'A7'] 收盘价
data.volumes  # pd.DataFrame 同形状（可能为 None，用前判空）
data.dates    # = data.prices.index
data.symbols  # = list(data.prices.columns)
data.n_assets # 资产数
data.periods_per_year  # 252
data.returns(periods=1)   # 收益率面板
data.rolling_vol(window)  # 年化滚动波动
```
合成数据性格：资产按索引轮换 trend(正自相关,利于动量) / meanrev(OU回归,利于均值回归) / random。
价格恒正。

## 策略契约
每个策略是 `Strategy` 子类，**必须能被无参构造**（`MyStrategy()`），参数写成类属性 `params` dict。
必填类属性：
```python
class MyStrategy(Strategy):
    name = "唯一_snake_case 名"      # 全局唯一，不能与已有策略重名
    channel = "<你的渠道名>"          # 与文件名一致，如 momentum
    universe = "timing" | "cross_section"
    long_only = True | False
    description = "一句话思路"
    hypothesis = "核心假设：为何可能有效/何时失效"
    source = "收集途径/灵感来源说明"
    params = {"k": v, ...}

    def generate_weights(self, data: MarketData) -> pd.DataFrame:
        # 返回 index=data.dates, columns=data.symbols 的权重面板
        ...
```
权重规则：
- 返回 DataFrame，形状 = (len(data.dates), len(data.symbols))，对齐 data 的 index/columns。
- 值域：`long_only=True` 时权重 ∈ [0,1]；可多空时 ∈ [-1,1]。**每行绝对值之和建议 ≤ 1**（不要加杠杆）。
- 逐资产择时(timing)：惯例把 0/1 信号乘以 `1/data.n_assets` 做等预算分配（参考 technical.py 的 `_to_weights`）。
- 截面(cross_section)：按因子排序给相对权重（如做多排名前 k、或按 rank 归一），每行和≈1 或≤1。
- 允许 NaN（会被引擎当 0），但请尽量 fillna。

可复用的内部小工具（如需状态机持有/等预算，自己在你的模块里实现，别 import technical 的私有函数）。

## 测试要求（tests/test_<渠道>.py，pytest，必须全绿）
至少覆盖：
- 你的每个策略：`generate_weights` 返回形状正确、列与 `data.symbols` 一致、数值有限。
- long_only 策略权重 ≥ 0；每行绝对值和 ≤ 1+eps。
- 确定性：同数据两次结果相同。
- 至少一个**行为断言**证明逻辑正确（如：构造上涨序列时动量策略持有；构造高/低波动时低波策略权重更高等）。
- 用 `kairos_strategies.make_synthetic_universe(n_assets=..., n_days=..., seed=...)` 或自造小 `MarketData` 造数据。

## 交付与自检
写完**务必自己运行**：
```bash
cd /home/zhuoming.wang/quant-hub/kairos/quant-strategies
python3 -m pytest -q tests/test_<你的渠道>.py     # 必须全绿
python3 -c "import kairos_strategies as ks; \
d=ks.make_synthetic_universe(n_assets=6,n_days=400,seed=1); \
[print(s.name, s.channel, s.generate_weights(d).shape) for s in ks.discover() if s.channel=='<你的渠道>']"
```
确保你的策略能被 `ks.discover()` 自动发现（类名/模块符合规范即可）。

## 渠道主题建议（按分配给你的渠道实现 3~5 个策略）
- momentum：时序动量(如过去 N 月符号)、截面动量(12-1)、52周新高接近度、双动量(绝对+相对)。
- meanrev：滚动 z-score 回归、布林回归、价差/配对(对两只资产做价差 z-score 多空)、移动平均偏离回归。
- factor：低波动异象、短期反转、趋势质量/价格效率、残差动量等（**仅用价格/波动构造**，不要臆造基本面）。
- volatility：波动率目标(按目标波动缩放敞口)、ATR 通道突破、波动率状态过滤(高波降杠杆)。
- crypto：网格(区间挂单等价权重)、定投(DCA 周期建仓)、24/7 动量、carry/资金费代理(用现货-远期价差概念，纯合成)。

最终只回复简报：文件清单、策略名列表、pytest 结果、一个行为断言示例、任何偏离。
