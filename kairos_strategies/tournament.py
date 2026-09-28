"""策略锦标赛 / 策略选择分析（QR 流程的「组合与优选」环节）。

在给定数据上回测全部策略 -> 得到各策略收益流 -> 计算相关性、层次聚类、
按夏普排序并做「去相关贪心精选」-> 构建分散化的策略组合(combo) -> 产出报告。

纯 numpy/pandas（scipy 可选用于层次聚类，缺失时回退贪心聚类）。离线、确定性。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import metrics
from .base import MarketData, Strategy
from .engine import Backtester


def strategy_returns(data: MarketData, strategies: Sequence[Strategy],
                     bt: Optional[Backtester] = None) -> Tuple[pd.DataFrame, List[str]]:
    """回测每个策略，返回 (收益矩阵 DataFrame[列=策略名], 失败策略名列表)。"""
    bt = bt or Backtester(cost_rate=0.001, periods_per_year=data.periods_per_year)
    cols, failed = {}, []
    for s in strategies:
        try:
            w = s.generate_weights(data)
            cols[s.name] = bt.run(data, w).returns
        except Exception:  # noqa: BLE001
            failed.append(s.name)
    return pd.DataFrame(cols), failed


def rank_by_sharpe(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """按夏普降序排名，附带常用指标。"""
    rows = []
    for name in returns.columns:
        r = returns[name]
        rows.append({
            "name": name,
            "sharpe": metrics.sharpe(r, periods_per_year=periods_per_year),
            "total_return": metrics.total_return(r),
            "cagr": metrics.cagr(r, periods_per_year),
            "volatility": metrics.volatility(r, periods_per_year),
            "max_drawdown": metrics.max_drawdown(r),
            "calmar": metrics.calmar(r, periods_per_year),
        })
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """相关矩阵 -> 距离矩阵 sqrt(0.5*(1-corr))，用于聚类。"""
    return np.sqrt(0.5 * (1.0 - corr.clip(-1, 1)))


def cluster_strategies(corr: pd.DataFrame, n_clusters: int = 6) -> Dict[str, List[str]]:
    """层次聚类（相关距离）。scipy 可用则用之，否则回退贪心聚类。返回 {簇标签: [策略名]}。"""
    names = list(corr.columns)
    dist = correlation_distance(corr)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        condensed = squareform(dist.values, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    except Exception:  # noqa: BLE001
        labels = _greedy_cluster(dist.values, n_clusters)
    out: Dict[str, List[str]] = {}
    for n, lab in zip(names, labels):
        out.setdefault(f"cluster_{int(lab)}", []).append(n)
    return out


def _greedy_cluster(D: np.ndarray, k: int) -> List[int]:
    """无 scipy 时的确定性贪心聚类：按到已选中心的距离分配。"""
    n = D.shape[0]
    centers = [int(np.argmax(D.sum(axis=1)))]  # 最"居中/分散"的点起步
    labels = [-1] * n
    labels[centers[0]] = 0
    while len(centers) < k:
        assigned = np.array([D[i, centers].min() if labels[i] >= 0 else np.inf for i in range(n)])
        assigned[[c for c in centers]] = -1
        nxt = int(np.nanargmax(np.where(np.isinf(assigned), -1, assigned)))
        if nxt in centers:
            break
        centers.append(nxt)
        labels[nxt] = len(centers) - 1
    for i in range(n):
        if labels[i] < 0:
            labels[i] = int(np.argmin(D[i, centers]))
    return labels


def select_diversified(returns: pd.DataFrame, top_k: int = 8, corr_thresh: float = 0.7,
                       periods_per_year: int = 252) -> List[str]:
    """去相关贪心精选：按夏普降序遍历，若与已选策略相关性 < 阈值则纳入。"""
    ranking = rank_by_sharpe(returns, periods_per_year)
    corr = returns.corr()
    picked: List[str] = []
    for name in ranking["name"]:
        if len(picked) >= top_k:
            break
        if all(abs(corr.loc[name, p]) < corr_thresh for p in picked):
            picked.append(name)
    return picked


def combo_returns(returns: pd.DataFrame, selected: Sequence[str],
                  method: str = "inverse_vol") -> pd.Series:
    """把选中策略的收益流合成为组合收益（等权 / 逆波动 / 夏普加权）。"""
    sub = returns[list(selected)]
    if method == "equal":
        w = pd.Series(1.0 / len(selected), index=selected)
    elif method == "sharpe":
        sh = sub.apply(lambda r: max(metrics.sharpe(r), 0.0))
        w = (sh / sh.sum()) if sh.sum() > 0 else pd.Series(1.0 / len(selected), index=selected)
    else:  # inverse_vol
        vol = sub.std().replace(0.0, np.nan)
        inv = 1.0 / vol
        w = (inv / inv.sum()).fillna(1.0 / len(selected))
    return (sub * w).sum(axis=1)


def _combo_metrics(r: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    zero = pd.Series(0.0, index=r.index)
    m = metrics.summarize(r, zero, periods_per_year=periods_per_year)
    m.pop("avg_turnover", None)
    m.pop("total_turnover", None)
    return m


def run_tournament(data: MarketData, strategies: Sequence[Strategy], out_root: str,
                   top_k: int = 8, corr_thresh: float = 0.7, n_clusters: int = 6,
                   combo_method: str = "inverse_vol", cost_rate: float = 0.001,
                   chart: bool = True) -> Dict:
    """跑完整锦标赛分析并落盘报告。返回结果字典。"""
    os.makedirs(out_root, exist_ok=True)
    bt = Backtester(cost_rate=cost_rate, periods_per_year=data.periods_per_year)
    returns, failed = strategy_returns(data, strategies, bt)
    ranking = rank_by_sharpe(returns, data.periods_per_year)
    corr = returns.corr()
    clusters = cluster_strategies(corr, n_clusters)
    selected = select_diversified(returns, top_k, corr_thresh, data.periods_per_year)
    combo = combo_returns(returns, selected, combo_method)
    combo_m = _combo_metrics(combo, data.periods_per_year)
    bench = returns["equal_weight_buy_hold"] if "equal_weight_buy_hold" in returns.columns else None
    bench_m = _combo_metrics(bench, data.periods_per_year) if bench is not None else None
    best = returns[ranking.iloc[0]["name"]]
    best_m = _combo_metrics(best, data.periods_per_year)

    # 落盘
    returns.to_csv(os.path.join(out_root, "strategy_returns.csv"))
    corr.round(4).to_csv(os.path.join(out_root, "correlation.csv"))
    ranking.to_csv(os.path.join(out_root, "ranking.csv"), index=False)
    (1.0 + combo).cumprod().to_frame("combo_equity").to_csv(os.path.join(out_root, "combo_equity.csv"))
    with open(os.path.join(out_root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"selected": selected, "corr_thresh": corr_thresh, "combo_method": combo_method,
                   "clusters": clusters, "failed": failed,
                   "combo_metrics": combo_m, "best_single": {"name": ranking.iloc[0]["name"], **best_m},
                   "benchmark": bench_m}, f, ensure_ascii=False, indent=2)
    if chart:
        try:
            from .report import save_chart
            save_chart((1.0 + combo).cumprod(), os.path.join(out_root, "combo_equity.png"),
                       "Tournament combo equity (diversified elite)")
        except Exception:  # noqa: BLE001
            pass
    with open(os.path.join(out_root, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(_report_md(data, ranking, clusters, selected, combo_m, best_m, bench_m,
                           ranking.iloc[0]["name"], combo_method, corr_thresh, len(failed)))
    return {"ranking": ranking, "selected": selected, "clusters": clusters,
            "combo_metrics": combo_m, "failed": failed}


def _report_md(data, ranking, clusters, selected, combo_m, best_m, bench_m,
               best_name, combo_method, corr_thresh, n_failed) -> str:
    def line(m):
        return (f"累计 {m['total_return']*100:.1f}% ｜ CAGR {m['cagr']*100:.1f}% ｜ 波动 {m['volatility']*100:.1f}% "
                f"｜ 夏普 {m['sharpe']:.2f} ｜ 索提诺 {m['sortino']:.2f} ｜ 回撤 {m['max_drawdown']*100:.1f}% "
                f"｜ 卡玛 {m['calmar']:.2f}")
    top = ranking.head(12)
    top_rows = "\n".join(
        f"| {i+1} | `{r.name}` | {r.sharpe:.2f} | {r.total_return*100:.1f}% | {r.volatility*100:.1f}% "
        f"| {r.max_drawdown*100:.1f}% | {r.calmar:.2f} |"
        for i, r in enumerate(top.itertuples()))
    cl = "\n".join(f"- **{k}** ({len(v)}): {', '.join('`'+x+'`' for x in sorted(v))}"
                   for k, v in sorted(clusters.items()))
    bench_line = f"\n- 等权买入持有基准：{line(bench_m)}" if bench_m else ""
    return f"""# 策略锦标赛报告 (Tournament Report)

> 数据：{data.name}（{data.n_assets} 资产，{len(data.dates)} 期）｜ 参与策略：{len(ranking)}（失败 {n_failed}）

对全部策略在同一数据上回测，得到各自收益流，再做**相关性聚类 + 去相关精选 + 分散化组合**，
目的是从众多策略中挑出一组**彼此低相关、各自较优**的「精英组合」，而非只追单一最高夏普。

## 夏普 Top 12
| # | 策略 | 夏普 | 累计 | 波动 | 回撤 | 卡玛 |
|---|---|---|---|---|---|---|
{top_rows}

## 相关性聚类（{len(clusters)} 簇）
按收益相关距离 sqrt(0.5·(1−ρ)) 层次聚类，同簇策略往往捕捉相似逻辑：

{cl}

## 去相关精选（corr < {corr_thresh}，取 {len(selected)} 个）
按夏普降序贪心纳入、跳过与已选高度相关者：
{', '.join('`'+s+'`' for s in selected)}

## 精英组合表现（{combo_method} 加权）
- **组合**：{line(combo_m)}
- 单一最佳（`{best_name}`）：{line(best_m)}{bench_line}

组合通过分散化通常能在**相近或更高夏普**下显著**降低回撤**——这正是策略组合的价值。

![combo](combo_equity.png)

## 复现
```bash
python examples/run_tournament.py            # 真实数据；或 --synthetic
```
产物：`strategy_returns.csv`、`correlation.csv`、`ranking.csv`、`selection.json`、`combo_equity.*`、本报告。

> 结果为历史数据演示，存在过拟合/幸存者偏差等局限，**不构成投资建议**。
"""
