# Kairos Strategies

> Kairos 量化系列的**策略研究库** —— 多渠道收集策略、100% 原创实现、向量化回测，
> 并把「策略 → 回测 → 结果」的完整 **QR（量化研究）流程**固化为仓库内的研究记录。

本项目不追求"圣杯"，而是提供一套**可复现、可扩展、可审计**的策略研究流水线：
每个策略都有明确的思路、来源途径、假设、参数、实现、回测设置与结果记录。

## 特性
- **多渠道策略收集**：按"来源途径"分渠道组织（技术分析 / 动量 / 均值回归 / 因子异象 / 波动率 / 加密 / 基准 …），`registry` 动态发现，新增渠道零侵入。
- **统一契约**：所有策略输出「目标权重面板」(date × asset)，引擎自动**滞后一期**执行，杜绝未来函数。
- **自研向量化回测**：换手×成本扣减、净值/指标一体化，纯 numpy/pandas。
- **研究记录固化**：`examples/run_all.py` 一键为每个策略生成 `README/result.json/equity.csv/equity.png` 与全局 `SUMMARY.md`。
- **可复现**：合成数据固定 seed、离线、确定性；也支持 `load_prices_csv` 跑真实数据。
- **自研指标库**：SMA/EMA/RSI/MACD/布林/唐奇安/ATR/滚动 z-score 等，供各渠道复用。

## QR 流程
完整流程见 [`research/QR_FLOW.md`](research/QR_FLOW.md)：
**收集 → 假设 → 实现 → 回测 → 评估 → 记录 → 复现 → 迭代**。

## 目录结构
```
kairos_strategies/
  base.py         策略契约 Strategy / 市场数据 MarketData / 权重对齐
  data.py         可复现合成 universe（趋势/均值回归/随机性格）+ CSV 加载
  indicators.py   自研技术指标库
  engine.py       向量化回测引擎（防未来函数 + 成本）
  metrics.py      绩效指标
  registry.py     策略动态发现（按渠道）
  report.py       研究记录与汇总生成
  channels/       各渠道策略（technical / benchmark / … 每个模块一类）
examples/run_all.py  跑通全流程并生成研究记录
research/
  QR_FLOW.md      QR 流程说明
  SUMMARY.md      所有策略回测汇总（自动生成）
  records/<name>/ 每个策略的研究记录（README/result.json/equity.csv/equity.png）
tests/            pytest 测试
```

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"     # plot 用于生成净值图（matplotlib）
```

## 快速开始
```bash
python examples/run_all.py       # 或 make research
```
运行后查看 `research/SUMMARY.md`（汇总表）与 `research/records/<策略>/`（逐策略记录）。

代码方式：
```python
import kairos_strategies as ks
data = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
for s in ks.discover():
    w = s.generate_weights(data)
    res = ks.Backtester(cost_rate=0.0005).run(data, w)
    print(s.name, round(res.metrics["sharpe"], 2))
```

## 策略渠道一览（106 策略 / 27 渠道）

| 渠道 | 策略数 | 策略 |
|---|---|---|
| `benchmark` | 1 | `equal_weight_buy_hold` |
| `technical` | 5 | `bollinger_breakout`, `donchian_turtle`, `macd_trend`, `rsi_reversion`, `sma_cross` |
| `trend` | 5 | `adx_trend`, `dual_thrust`, `ma_ribbon`, `tsmom_volscaled`, `turtle_atr` |
| `breakout` | 4 | `channel_atr_breakout`, `keltner_breakout`, `range_breakout`, `volatility_breakout` |
| `momentum` | 4 | `dual_momentum`, `high_52w`, `ts_momentum`, `xs_momentum` |
| `momentum_adv` | 4 | `frog_in_pan`, `group_momentum`, `momentum_spread_timing`, `vol_managed_momentum` |
| `meanrev` | 4 | `bollinger_reversion`, `ma_deviation`, `pairs_spread`, `zscore_reversion` |
| `factor` | 4 | `idio_momentum`, `low_volatility`, `short_term_reversal`, `trend_quality` |
| `multi_factor` | 4 | `equalweight_composite`, `ic_weighted_composite`, `max_ir_composite`, `pca_factor` |
| `allocation` | 5 | `equal_weight_rebal`, `inverse_vol`, `max_diversification`, `min_variance_alloc`, `risk_parity_alloc` |
| `hrp` | 3 | `clustered_inverse_vol`, `herc`, `hrp` |
| `statarb` | 4 | `basket_neutral`, `coint_pairs`, `eof_stat_arb`, `xs_zscore_reversion` |
| `pairs` | 3 | `coint_pairs_portfolio`, `sector_neutral_pairs`, `ssd_pairs` |
| `long_short` | 5 | `lowvol_ls`, `quality_ls`, `residual_momentum_ls`, `reversal_ls`, `xs_momentum_ls` |
| `market_making` | 4 | `avellaneda_stoikov_proxy`, `grid_mm_daily`, `inventory_skew_mm`, `liquidity_provision_ls` |
| `dispersion` | 3 | `correlation_regime`, `dispersion_timing`, `vol_dispersion_ls` |
| `volatility` | 4 | `atr_breakout`, `vol_regime_filter`, `vol_scaled_momentum`, `vol_target` |
| `regime` | 4 | `beta_timing`, `em_regime`, `regime_vol_timing`, `trend_regime_switch` |
| `taa` | 4 | `dual_momentum_taa`, `mom_12m_taa`, `trend_regime_taa`, `vol_target_taa` |
| `riskmgmt` | 5 | `circuit_breaker`, `cppi`, `drawdown_throttle`, `portfolio_vol_target`, `trailing_stop_overlay` |
| `defensive` | 4 | `betting_against_beta`, `defensive_quality`, `drawdown_averse`, `low_beta_timing` |
| `ml_signal` | 4 | `gbm_alpha`, `knn_alpha`, `logit_signal`, `ridge_alpha` |
| `adaptive` | 4 | `adaptive_param`, `bandit_select`, `hedge_experts`, `online_mom_rev_switch` |
| `ensemble` | 4 | `equal_weight_ensemble`, `inverse_vol_ensemble`, `sharpe_weighted_ensemble`, `trend_reversion_blend` |
| `microstructure` | 4 | `liquidity_premium`, `volume_imbalance`, `volume_price_divergence`, `vwap_reversion` |
| `seasonality` | 3 | `month_of_year`, `turn_of_month`, `weekday_effect` |
| `crypto` | 4 | `carry_proxy`, `crypto_momentum_247`, `dca`, `grid_trading` |

完整回测指标见 [`research/SUMMARY.md`](research/SUMMARY.md)；样本外验证见 [`research/VALIDATION_SUMMARY.md`](research/VALIDATION_SUMMARY.md)；逐策略研究记录在 `research/records/<name>/`。


## 新增一个策略（贡献指南）
1. 在 `kairos_strategies/channels/<渠道>.py` 中定义一个 `Strategy` 子类；
2. 填写元信息（`name`/`channel`/`description`/`hypothesis`/`source`/`params`）；
3. 实现 `generate_weights(data) -> 权重面板`（只用截至当期信息，long_only 则权重≥0）；
4. 加测试；运行 `python examples/run_all.py` 生成记录。
详见 `CHANNEL_SPEC.md`。

## 真实数据回测（A 股）
除合成数据外，本库支持在**真实 A 股行情**上回测全部策略：

```bash
python examples/run_real.py            # 无本地数据则自动联网抓取(腾讯公开行情)
python examples/run_real.py --data-dir /path/to/csv --cost 0.001
```
- 真实数据加载/抓取见 `kairos_strategies/realdata.py`（自包含；自动处理停牌、上市日、非正价、防未来）。
- 默认股票池：38 只跨行业流动 A 股，约 8 年（2018-10 ~ 2026-09）前复权日线。
- 产物：`research/real/records/<策略>/`（逐策略真实回测记录）+ `research/real/SUMMARY.md`（真实数据汇总）+ `REAL_NOTES.md`。
- 合成数据结果在 `research/SUMMARY.md`，真实数据结果在 `research/real/SUMMARY.md`，便于对照。
- **行业均衡回测**：`python examples/run_sector.py` 对每个策略叠加 `sector_equalize`（8 行业分组，消除行业集中度），结果在 `research/real_sector/`。
- **真实数据样本外验证**：`python examples/validate_real.py` 在真实 A 股上产出 walk-forward OOS / 参数敏感性 / PSR / bootstrap CI，结果在 `research/real/validation/` 与 `research/real/VALIDATION_SUMMARY.md`。

> 真实数据来自公开行情接口，仅用于研究演示；结果存在过拟合/幸存者偏差/样本区间依赖等局限，**不构成投资建议**。

## 数据与免责声明
- 默认使用**合成数据**（固定 seed、离线、可复现），植入趋势/均值回归/随机三种"市场性格"，
  仅用于验证实现正确性与演示 QR 流程。
- 所有回测结果**不构成任何投资建议或收益承诺**；真实使用前请在自有数据上重跑并做参数敏感性/样本外验证。

## 样本外验证 (Validation)

`examples/validate.py` 对每个策略产出稳健性证据（写入 `research/validation/<name>/`）：
- **Walk-forward OOS**：连续样本外窗口的收益/夏普与一致性（正收益窗口占比）。
- **参数敏感性**：对数值参数做 OAT 扰动，给出夏普的 min/median/max（越窄越稳健）。
- **概率夏普比率 PSR**：按偏度/峰度校正，衡量夏普显著性。
- **Block-bootstrap 置信区间**：移动块自助法给出年化夏普的 95% CI（保留自相关）。

```bash
python examples/validate.py        # quick 模式；--full 更细
```

## 测试
```bash
make test          # 或 python -m pytest -q
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。策略思路来源于公开、通用的量化范式
（技术分析、时序/截面动量、均值回归与统计套利、因子异象、波动率目标、加密市场实践等），
在此向开源量化社区与相关经典文献致谢；所有算法、接口与实现均为本仓库自研。
