# 策略锦标赛报告 (Tournament Report)

> 数据：ashare_real（38 资产，1931 期）｜ 参与策略：106（失败 0）

对全部策略在同一数据上回测，得到各自收益流，再做**相关性聚类 + 去相关精选 + 分散化组合**，
目的是从众多策略中挑出一组**彼此低相关、各自较优**的「精英组合」，而非只追单一最高夏普。

## 夏普 Top 12
| # | 策略 | 夏普 | 累计 | 波动 | 回撤 | 卡玛 |
|---|---|---|---|---|---|---|
| 1 | `trailing_stop_overlay` | 0.96 | 113.1% | 10.9% | 14.7% | 0.71 |
| 2 | `range_breakout` | 0.87 | 63.1% | 7.7% | 14.0% | 0.47 |
| 3 | `turn_of_month` | 0.85 | 62.6% | 7.8% | 12.1% | 0.54 |
| 4 | `donchian_turtle` | 0.84 | 74.1% | 9.1% | 14.3% | 0.52 |
| 5 | `turtle_atr` | 0.83 | 66.7% | 8.5% | 13.9% | 0.49 |
| 6 | `sma_cross` | 0.81 | 85.8% | 10.7% | 13.8% | 0.61 |
| 7 | `carry_proxy` | 0.80 | 84.0% | 10.7% | 14.9% | 0.55 |
| 8 | `equal_weight_rebal` | 0.79 | 154.2% | 17.5% | 30.7% | 0.42 |
| 9 | `atr_breakout` | 0.78 | 69.9% | 9.4% | 17.8% | 0.40 |
| 10 | `channel_atr_breakout` | 0.78 | 64.5% | 8.8% | 15.4% | 0.44 |
| 11 | `volume_imbalance` | 0.77 | 86.2% | 11.4% | 19.8% | 0.43 |
| 12 | `drawdown_throttle` | 0.77 | 86.5% | 11.5% | 29.9% | 0.28 |

## 相关性聚类（6 簇）
按收益相关距离 sqrt(0.5·(1−ρ)) 层次聚类，同簇策略往往捕捉相似逻辑：

- **cluster_1** (85): `adaptive_param`, `adx_trend`, `atr_breakout`, `bandit_select`, `beta_timing`, `bollinger_breakout`, `carry_proxy`, `channel_atr_breakout`, `circuit_breaker`, `clustered_inverse_vol`, `correlation_regime`, `cppi`, `crypto_momentum_247`, `dca`, `defensive_quality`, `dispersion_timing`, `donchian_turtle`, `drawdown_averse`, `drawdown_throttle`, `dual_momentum`, `dual_momentum_taa`, `dual_thrust`, `em_regime`, `equal_weight_buy_hold`, `equal_weight_ensemble`, `equal_weight_rebal`, `equalweight_composite`, `frog_in_pan`, `gbm_alpha`, `grid_trading`, `group_momentum`, `hedge_experts`, `herc`, `high_52w`, `hrp`, `ic_weighted_composite`, `idio_momentum`, `inverse_vol`, `inverse_vol_ensemble`, `keltner_breakout`, `liquidity_premium`, `logit_signal`, `low_beta_timing`, `low_volatility`, `ma_ribbon`, `macd_trend`, `max_diversification`, `max_ir_composite`, `min_variance_alloc`, `mom_12m_taa`, `momentum_spread_timing`, `month_of_year`, `online_mom_rev_switch`, `pca_factor`, `portfolio_vol_target`, `range_breakout`, `regime_vol_timing`, `residual_momentum_ls`, `ridge_alpha`, `risk_parity_alloc`, `rsi_reversion`, `sharpe_weighted_ensemble`, `short_term_reversal`, `sma_cross`, `trailing_stop_overlay`, `trend_quality`, `trend_regime_switch`, `trend_regime_taa`, `trend_reversion_blend`, `ts_momentum`, `tsmom_volscaled`, `turn_of_month`, `turtle_atr`, `vol_managed_momentum`, `vol_regime_filter`, `vol_scaled_momentum`, `vol_target`, `vol_target_taa`, `volatility_breakout`, `volume_imbalance`, `volume_price_divergence`, `vwap_reversion`, `weekday_effect`, `xs_momentum`, `xs_momentum_ls`
- **cluster_2** (1): `knn_alpha`
- **cluster_3** (4): `betting_against_beta`, `lowvol_ls`, `quality_ls`, `vol_dispersion_ls`
- **cluster_4** (14): `avellaneda_stoikov_proxy`, `basket_neutral`, `bollinger_reversion`, `coint_pairs_portfolio`, `eof_stat_arb`, `grid_mm_daily`, `inventory_skew_mm`, `liquidity_provision_ls`, `ma_deviation`, `pairs_spread`, `reversal_ls`, `sector_neutral_pairs`, `xs_zscore_reversion`, `zscore_reversion`
- **cluster_5** (1): `ssd_pairs`
- **cluster_6** (1): `coint_pairs`

## 去相关精选（corr < 0.7，取 8 个）
按夏普降序贪心纳入、跳过与已选高度相关者：
`trailing_stop_overlay`, `range_breakout`, `turn_of_month`, `risk_parity_alloc`, `mom_12m_taa`, `idio_momentum`, `rsi_reversion`, `xs_momentum_ls`

## 精英组合表现（inverse_vol 加权）
- **组合**：累计 64.9% ｜ CAGR 6.7% ｜ 波动 7.3% ｜ 夏普 0.93 ｜ 索提诺 1.40 ｜ 回撤 15.2% ｜ 卡玛 0.44
- 单一最佳（`trailing_stop_overlay`）：累计 113.1% ｜ CAGR 10.4% ｜ 波动 10.9% ｜ 夏普 0.96 ｜ 索提诺 1.44 ｜ 回撤 14.7% ｜ 卡玛 0.71
- 等权买入持有基准：累计 145.3% ｜ CAGR 12.4% ｜ 波动 17.4% ｜ 夏普 0.76 ｜ 索提诺 1.12 ｜ 回撤 31.2% ｜ 卡玛 0.40

组合通过分散化通常能在**相近或更高夏普**下显著**降低回撤**——这正是策略组合的价值。

![combo](combo_equity.png)

## 复现
```bash
python examples/run_tournament.py            # 真实数据；或 --synthetic
```
产物：`strategy_returns.csv`、`correlation.csv`、`ranking.csv`、`selection.json`、`combo_equity.*`、本报告。

> 结果为历史数据演示，存在过拟合/幸存者偏差等局限，**不构成投资建议**。
