"""把「策略 -> 回测 -> 结果」固化为仓库内的研究记录（QR 流程产物）。

每个策略生成：
  research/records/<name>/README.md    研究记录（思路/来源途径/假设/参数/实现/数据/设置/结果/结论）
  research/records/<name>/result.json  指标与元信息
  research/records/<name>/equity.csv   净值曲线
  research/records/<name>/equity.png   净值图（matplotlib 可用时）
并汇总 research/SUMMARY.md。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import MarketData, Strategy
from .engine import Backtester, BacktestResult


def _fmt(v: float, pct: bool = False, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
        return "inf" if v > 0 else ("nan" if np.isnan(v) else "-inf")
    return f"{v*100:.2f}%" if pct else f"{v:.{nd}f}"


def save_chart(equity: pd.Series, path: str, title: str) -> bool:
    """保存净值曲线 PNG（英文标签，避免中文字体缺失）。失败返回 False。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    try:
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
        ax.plot(equity.index, equity.values, lw=1.2)
        ax.set_title(title)
        ax.set_ylabel("Equity (start=1.0)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return True
    except Exception:
        return False


def record_readme(s: Strategy, m: Dict[str, float], data: MarketData,
                  bt: Backtester, has_chart: bool) -> str:
    meta = s.meta()
    params = "\n".join(f"- `{k}` = {v}" for k, v in meta["params"].items()) or "- （无）"
    span = f"{data.dates[0].date()} ~ {data.dates[-1].date()}" if len(data.dates) else "NA"
    chart = "![equity](equity.png)\n" if has_chart else "_（未生成图表）_\n"
    return f"""# 策略研究记录：{meta['name']}

> 类别/途径：**{meta['channel']}** ｜ 类型：{meta['universe']} ｜ 方向：{'只做多' if meta['long_only'] else '可多空'}

## 一句话思路
{meta['description'] or '（待补）'}

## 收集途径 / 灵感来源
{meta['source'] or '（待补）'}

## 核心假设
{meta['hypothesis'] or '（待补）'}

## 参数
{params}

## 实现要点
- 实现文件：`kairos_strategies/channels/{meta['channel']}.py`（类名对应本策略）。
- 输出统一为「目标权重面板」(date × asset)，由 `Backtester` 滞后一期执行，杜绝未来函数。
- 仅使用 numpy/pandas，离线、确定性。

## 数据与回测设置
| 项 | 值 |
|---|---|
| 数据集 | {data.name}（{data.n_assets} 资产） |
| 区间 | {span} |
| 期数 | {int(m['n_periods'])} |
| 年化基准期数 | {bt.periods_per_year} |
| 单边成本率 | {bt.cost_rate:.4%} |
| 无风险利率 | {bt.risk_free:.2%} |

## 回测结果
| 指标 | 值 |
|---|---|
| 累计收益 | {_fmt(m['total_return'], pct=True)} |
| 年化收益 (CAGR) | {_fmt(m['cagr'], pct=True)} |
| 年化波动 | {_fmt(m['volatility'], pct=True)} |
| 夏普 | {_fmt(m['sharpe'])} |
| 索提诺 | {_fmt(m['sortino'])} |
| 最大回撤 | {_fmt(m['max_drawdown'], pct=True)} |
| 卡玛 | {_fmt(m['calmar'])} |
| 胜率 | {_fmt(m['win_rate'], pct=True)} |
| 盈利因子 | {_fmt(m['profit_factor'])} |
| 平均换手 | {_fmt(m['avg_turnover'])} |
| 累计换手 | {_fmt(m['total_turnover'], nd=2)} |

## 净值曲线
{chart}
## 结论与改进方向
- 结果为 **{"合成数据演示，用于验证实现正确性与 QR 流程" if str(data.name).startswith("synthetic") else f"真实历史数据（{data.name}）回测演示，存在过拟合/幸存者偏差/样本区间依赖等局限"}**，不构成任何投资建议或收益承诺。
- 改进方向：在真实数据上重跑、参数敏感性分析、加入波动率目标/风控叠加、与其它策略做相关性分散。
"""


def summary_md(rows: List[Dict], data: MarketData, bt: Backtester) -> str:
    """生成所有策略的汇总表（按渠道分组，按夏普排序）。"""
    lines = [
        "# 策略回测汇总 (SUMMARY)",
        "",
        f"> 数据集：{data.name}（{data.n_assets} 资产，{int(rows[0]['n_periods']) if rows else 0} 期）  ",
        f"> 回测：向量化、权重滞后一期、单边成本 {bt.cost_rate:.4%}。结果由回测流水线自动生成，**仅供研究，非投资建议**。",
        "",
    ]
    by_ch: Dict[str, List[Dict]] = {}
    for r in rows:
        by_ch.setdefault(r["channel"], []).append(r)
    for ch in sorted(by_ch):
        lines.append(f"## {ch}")
        lines.append("")
        lines.append("| 策略 | 累计收益 | CAGR | 波动 | 夏普 | 索提诺 | 最大回撤 | 卡玛 | 胜率 | 累计换手 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(by_ch[ch], key=lambda x: x["sharpe"], reverse=True):
            lines.append(
                f"| [{r['name']}](records/{r['name']}) "
                f"| {_fmt(r['total_return'], True)} | {_fmt(r['cagr'], True)} "
                f"| {_fmt(r['volatility'], True)} | {_fmt(r['sharpe'])} | {_fmt(r['sortino'])} "
                f"| {_fmt(r['max_drawdown'], True)} | {_fmt(r['calmar'])} "
                f"| {_fmt(r['win_rate'], True)} | {_fmt(r['total_turnover'], nd=2)} |"
            )
        lines.append("")
    lines.append(f"合计策略数：**{len(rows)}**。逐策略详情见 `records/<name>/`。")
    return "\n".join(lines) + "\n"


def run_strategy(s: Strategy, data: MarketData, bt: Backtester, records_root: str,
                 chart: bool = True) -> Dict:
    """回测单个策略并落盘研究记录，返回汇总行。"""
    w = s.generate_weights(data)
    res: BacktestResult = bt.run(data, w)
    m = dict(res._metrics)
    d = os.path.join(records_root, s.name)
    os.makedirs(d, exist_ok=True)
    has_chart = False
    if chart:
        has_chart = save_chart(res.equity, os.path.join(d, "equity.png"), f"{s.name} equity curve")
    pd.DataFrame({"equity": res.equity}).to_csv(os.path.join(d, "equity.csv"))
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": s.meta(), "metrics": m,
                   "data": data.name, "cost_rate": bt.cost_rate}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(record_readme(s, m, data, bt, has_chart))
    row = {"name": s.name, "channel": s.channel, **m}
    return row


def run_all(data: MarketData, strategies: List[Strategy], research_root: str,
            cost_rate: float = 0.001, chart: bool = True) -> List[Dict]:
    """回测全部策略，写各自记录 + 汇总 SUMMARY.md，返回汇总行列表。"""
    bt = Backtester(cost_rate=cost_rate, periods_per_year=data.periods_per_year)
    records_root = os.path.join(research_root, "records")
    os.makedirs(records_root, exist_ok=True)
    rows = [run_strategy(s, data, bt, records_root, chart=chart) for s in strategies]
    with open(os.path.join(research_root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(summary_md(rows, data, bt))
    return rows
