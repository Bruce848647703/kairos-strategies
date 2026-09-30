"""压力 / 情景稳健性测试：把全部策略丢进多种「合成市场情景」，看谁扛得住。

与 `run_all.py`（单一混合性格 universe）互补：这里每个情景只有一种极端/不利性格
（单边上涨、崩盘、泡沫、震荡、高波、风格轮动），用来回答两个问题——
① 谁在**最差情景**下还能活（worst-case 夏普）；② 谁只是**顺风局选手**（只在单边行情有效）。

运行：
  python examples/run_stress.py                                  # 全部情景，1000 期
  python examples/run_stress.py --n-days 500                     # 小规模快速验证
  python examples/run_stress.py --kinds crash,bubble,high_vol --cost 0.002
产物： research/stress/{stress_metrics.csv, STRESS_SUMMARY.md}

全程离线、确定性；数据由 kairos_strategies/scenarios.py 合成，不读写任何已有 research 产物。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import kairos_strategies as ks
from kairos_strategies import metrics
from kairos_strategies.base import MarketData, Strategy
from kairos_strategies.engine import Backtester
from kairos_strategies.scenarios import SCENARIOS, describe_scenario, make_scenario

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRIC_COLS = ["scenario", "strategy", "channel", "sharpe", "total_return", "max_drawdown"]
STRESS_KINDS = ["crash", "bubble", "high_vol"]     # 「压力情景」子集，用于贴抗压标签


def buy_hold_metrics(data: MarketData) -> Dict[str, float]:
    """等权买入持有基准（期初等权、不再平衡），作为各情景的对照。"""
    eq = (data.prices / data.prices.iloc[0]).mean(axis=1)
    r = eq.pct_change().fillna(0.0)
    zero = pd.Series(0.0, index=r.index)
    return metrics.summarize(r, zero, periods_per_year=data.periods_per_year)


def run_scenario(kind: str, strategies: Sequence[Strategy], data: MarketData,
                 cost_rate: float) -> Tuple[List[Dict], List[str]]:
    """在单个情景上回测全部策略，返回 (指标行列表, 失败策略列表)。失败者跳过不中断。"""
    bt = Backtester(cost_rate=cost_rate, periods_per_year=data.periods_per_year)
    rows, failed = [], []
    for s in strategies:
        try:
            m = bt.run(data, s.generate_weights(data)).metrics
            rows.append({"scenario": kind, "strategy": s.name, "channel": s.channel,
                         "sharpe": m["sharpe"], "total_return": m["total_return"],
                         "max_drawdown": m["max_drawdown"]})
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{s.name}({type(exc).__name__})")
    return rows, failed


def run_stress(strategies: Sequence[Strategy], kinds: Sequence[str], n_days: int,
               n_assets: int, seed: int, cost_rate: float
               ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """逐情景回测全部策略：返回 (长表, 各情景买入持有基准, 各情景失败清单)。"""
    rows, bench, failed = [], {}, {}
    for i, kind in enumerate(kinds, 1):
        t0 = time.time()
        data = make_scenario(kind, n_assets=n_assets, n_days=n_days, seed=seed)
        bench[kind] = buy_hold_metrics(data)
        part, bad = run_scenario(kind, strategies, data, cost_rate)
        rows.extend(part)
        failed[kind] = bad
        b = bench[kind]
        print(f"  [{i}/{len(kinds)}] {kind:<16} {len(data.dates)}期×{data.n_assets}资产 "
              f"｜ 成功 {len(part)}/{len(strategies)} ｜ 失败 {len(bad)} "
              f"｜ 基准夏普 {b['sharpe']:+.2f} 回撤 {b['max_drawdown'] * 100:.1f}% "
              f"｜ {time.time() - t0:.1f}s")
    return pd.DataFrame(rows), bench, failed


def robustness_table(long_df: pd.DataFrame, kinds: Sequence[str]
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """长表 -> (稳健性汇总表, 跨情景夏普矩阵)。

    汇总列：各情景夏普 + n_scenarios / worst_sharpe / best_sharpe / mean_sharpe /
    n_positive(正收益情景数) / worst_scenario / worst_return / max_dd_worst / spread。
    排序：worst_sharpe 降序 -> n_positive 降序 -> mean_sharpe 降序。
    """
    sharpe = long_df.pivot(index="strategy", columns="scenario", values="sharpe").reindex(columns=list(kinds))
    ret = long_df.pivot(index="strategy", columns="scenario", values="total_return").reindex(columns=list(kinds))
    mdd = long_df.pivot(index="strategy", columns="scenario", values="max_drawdown").reindex(columns=list(kinds))
    channel = long_df.drop_duplicates("strategy").set_index("strategy")["channel"]

    out = pd.DataFrame(index=sharpe.index)
    out["channel"] = channel.reindex(sharpe.index)
    for k in kinds:
        out[k] = sharpe[k]
    out["n_scenarios"] = sharpe.notna().sum(axis=1)
    out["worst_sharpe"] = sharpe.min(axis=1)
    out["best_sharpe"] = sharpe.max(axis=1)
    out["mean_sharpe"] = sharpe.mean(axis=1)
    out["spread"] = out["best_sharpe"] - out["worst_sharpe"]
    out["n_positive"] = (ret > 0).sum(axis=1)
    out["worst_scenario"] = sharpe.idxmin(axis=1)
    out["worst_return"] = ret.min(axis=1)
    out["max_dd_worst"] = mdd.max(axis=1)
    out = out.sort_values(["worst_sharpe", "n_positive", "mean_sharpe"],
                          ascending=[False, False, False])
    return out, sharpe


def classify(robust: pd.DataFrame, kinds: Sequence[str],
             bench: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    """按跨情景表现贴标签（互斥，按优先级判定）：

    all_weather       全天候：最差情景夏普 > 0 且所有情景都为正收益
    stress_resilient  抗压型：崩盘/泡沫/高波三个压力情景**都不输给「什么都不做」基准**，
                      且最差情景夏普 > -0.6
    fragile           脆弱型：最差情景夏普 <= -1.5（某个情景下直接崩溃）
    trend_dependent   单边依赖型：trend_up 夏普 >= 1（顺风局很漂亮）但存在负夏普情景
    mixed             其它/混合型
    """
    stress = [k for k in STRESS_KINDS if k in robust.columns and k in kinds]
    has_trend = "trend_up" in robust.columns
    labels: Dict[str, List[str]] = {"all_weather": [], "stress_resilient": [], "fragile": [],
                                    "trend_dependent": [], "mixed": []}
    for name, row in robust.iterrows():
        worst = row["worst_sharpe"]
        if pd.isna(worst):
            labels["mixed"].append(name)
            continue
        beat_stress = bool(stress) and all(
            pd.notna(row[k]) and row[k] >= bench[k]["sharpe"] for k in stress)
        trend = row["trend_up"] if has_trend else float("nan")
        if worst > 0 and row["n_positive"] == row["n_scenarios"]:
            labels["all_weather"].append(name)
        elif beat_stress and worst > -0.6:
            labels["stress_resilient"].append(name)
        elif worst <= -1.5:
            labels["fragile"].append(name)
        elif pd.notna(trend) and trend >= 1.0 and worst < 0:
            labels["trend_dependent"].append(name)
        else:
            labels["mixed"].append(name)
    return labels


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _fmt_names(names: Sequence[str], limit: int = 12) -> str:
    if not names:
        return "（无）"
    head = ", ".join(f"`{n}`" for n in names[:limit])
    return head + (f" 等 {len(names)} 个" if len(names) > limit else "")


def build_summary_md(long_df: pd.DataFrame, robust: pd.DataFrame, sharpe: pd.DataFrame,
                     bench: Dict[str, Dict[str, float]], failed: Dict[str, List[str]],
                     kinds: Sequence[str], meta: Dict) -> str:
    """生成 STRESS_SUMMARY.md 文本。"""
    labels = classify(robust, kinds, bench)
    n_fail = sum(len(v) for v in failed.values())
    lines: List[str] = [
        "# 压力 / 情景稳健性测试报告 (Stress & Scenario Robustness)",
        "",
        "> ⚠️ **全部行情为合成情景**（`kairos_strategies/scenarios.py` 离线生成，固定 seed 可复现），"
        "仅用于演示策略在不同「市场性格」下的**稳健性**，"
        "**不构成任何投资建议**，也不代表任何真实资产的历史或未来表现。",
        "",
        f"> 情景 {len(kinds)} 种 ｜ 策略 {meta['n_strategies']} 个（成功 {meta['n_ok']} 条回测，失败 {n_fail}）"
        f"｜ 每情景 {meta['n_assets']} 资产 × {meta['n_days']} 期 ｜ 单边成本 {meta['cost_rate']:.3%} ｜ seed {meta['seed']}",
        "",
        "## 情景设定与买入持有基准",
        "",
        _md_row(["情景", "市场性格（合成植入）", "基准累计", "基准夏普", "基准回撤"]),
        _md_row(["---"] * 5),
    ]
    for k in kinds:
        b = bench[k]
        lines.append(_md_row([f"`{k}`", describe_scenario(k), f"{b['total_return'] * 100:+.1f}%",
                              f"{b['sharpe']:+.2f}", f"{b['max_drawdown'] * 100:.1f}%"]))
    lines += ["", "> 基准 = 等权买入持有（期初等权、不再平衡）。它是每个情景里「什么都不做」的成绩，",
              "> 策略只有跑赢它才算在该情景下创造了价值。", "",
              "## 一、各情景表现（夏普 Top5 / Bottom3）", ""]
    for i, k in enumerate(kinds, 1):
        sub = long_df[long_df["scenario"] == k].sort_values("sharpe", ascending=False)
        b = bench[k]
        lines += [
            f"### {i}. `{k}` — {describe_scenario(k)}",
            "",
            f"> 基准：累计 {b['total_return'] * 100:+.1f}% ｜ 夏普 {b['sharpe']:+.2f} ｜ 回撤 {b['max_drawdown'] * 100:.1f}%",
            "",
            "**夏普 Top5**",
            "",
            _md_row(["#", "策略", "渠道", "夏普", "累计", "回撤"]),
            _md_row(["---"] * 6),
        ]
        for j, r in enumerate(sub.head(5).itertuples(), 1):
            lines.append(_md_row([str(j), f"`{r.strategy}`", r.channel, f"{r.sharpe:+.2f}",
                                  f"{r.total_return * 100:+.1f}%", f"{r.max_drawdown * 100:.1f}%"]))
        lines += ["", "**夏普 Bottom3**", "",
                  _md_row(["#", "策略", "渠道", "夏普", "累计", "回撤"]),
                  _md_row(["---"] * 6)]
        tail = sub.tail(3).iloc[::-1]
        for j, r in enumerate(tail.itertuples(), 1):
            lines.append(_md_row([str(j), f"`{r.strategy}`", r.channel, f"{r.sharpe:+.2f}",
                                  f"{r.total_return * 100:+.1f}%", f"{r.max_drawdown * 100:.1f}%"]))
        if failed.get(k):
            lines += ["", f"> 本情景回测失败并跳过：{', '.join('`' + x + '`' for x in failed[k])}"]
        lines.append("")

    lines += ["## 二、跨情景稳健性矩阵（值 = 夏普，按最差情景降序）", "",
              "> 一行看完一个策略在**所有性格**下的表现：横向越平（spread 小）越稳健，",
              "> 某一列特别高而其它列为负 = 典型的「看天吃饭」。", "",
              _md_row(["策略", "渠道"] + [f"`{k}`" for k in kinds] +
                      ["worst", "mean", "spread", "正收益情景"]),
              _md_row(["---"] * (len(kinds) + 6))]
    for name, row in robust.iterrows():
        cells = [f"`{name}`", row["channel"]]
        cells += [f"{row[k]:+.2f}" if pd.notna(row[k]) else "—" for k in kinds]
        cells += [f"**{row['worst_sharpe']:+.2f}**", f"{row['mean_sharpe']:+.2f}",
                  f"{row['spread']:.2f}", f"{int(row['n_positive'])}/{int(row['n_scenarios'])}"]
        lines.append(_md_row(cells))

    top = robust.head(10)
    lines += ["", "## 三、稳健性排行 Top10（worst-case 优先）", "",
              "> 排序口径：**最差情景夏普(worst-case)** ↓ → **正收益情景数** ↓ → 平均夏普 ↓。",
              "> worst-case 是「最坏情况下我会怎样」的答案，比单看最高夏普更能反映可持有性。", "",
              _md_row(["#", "策略", "渠道", "最差夏普", "最差情景", "正收益情景", "平均夏普",
                       "最差累计", "最深回撤"]),
              _md_row(["---"] * 9)]
    for i, (name, row) in enumerate(top.iterrows(), 1):
        lines.append(_md_row([str(i), f"`{name}`", row["channel"], f"**{row['worst_sharpe']:+.2f}**",
                              f"`{row['worst_scenario']}`",
                              f"{int(row['n_positive'])}/{int(row['n_scenarios'])}",
                              f"{row['mean_sharpe']:+.2f}", f"{row['worst_return'] * 100:+.1f}%",
                              f"{row['max_dd_worst'] * 100:.1f}%"]))

    def best_in(kind: str, n: int = 3) -> str:
        if kind not in sharpe.columns:
            return "（未运行）"
        s = sharpe[kind].dropna().sort_values(ascending=False).head(n)
        return ", ".join(f"`{k}`({v:+.2f})" for k, v in s.items()) or "（无）"

    lines += ["", "## 四、结论：谁抗压，谁只是顺风", "",
              f"- **全天候**（最差情景夏普 > 0 且所有情景均正收益，{len(labels['all_weather'])} 个）："
              f"{_fmt_names(labels['all_weather'])}",
              f"- **抗压型**（崩盘/泡沫/高波下都不输基准，且最差情景夏普 > -0.6，{len(labels['stress_resilient'])} 个）："
              f"{_fmt_names(labels['stress_resilient'])} —— 能上榜的通常是**低波 / 风险预算配置**与"
              "**带风控闸门**（止损、波动率目标、回撤节流）的策略：它们不预测方向，"
              "靠压低敞口天然抗跌，因此崩盘与高波下仍能守住正夏普或仅小幅跑输。",
              f"- **单边依赖型**（`trend_up` 夏普 ≥ 1，但存在负夏普情景，{len(labels['trend_dependent'])} 个）："
              f"{_fmt_names(labels['trend_dependent'])} —— 这类策略的收益本质是**行情贝塔**，"
              "换个性格就失效，不可只看顺风局夏普上仓位。",
              f"- **脆弱型**（最差情景夏普 ≤ -1.5，某个情景下直接崩溃，{len(labels['fragile'])} 个）："
              f"{_fmt_names(labels['fragile'])}",
              f"- **混合型**（不属于以上任一类，{len(labels['mixed'])} 个）：{_fmt_names(labels['mixed'])}",
              "",
              f"> 抗压型的判定口径：在压力情景（{', '.join('`' + k + '`' for k in STRESS_KINDS if k in kinds)}）中，"
              "夏普**都不低于该情景的等权买入持有基准**（即至少不输给「什么都不做」）。",
              "",
              "**各压力情景下最能打的策略：**",
              f"- 崩盘 `crash`：{best_in('crash')}",
              f"- 泡沫破裂 `bubble`：{best_in('bubble')}",
              f"- 高波动 `high_vol`：{best_in('high_vol')}",
              f"- 区间震荡 `choppy`：{best_in('choppy')}",
              f"- 单边上涨 `trend_up`：{best_in('trend_up')}",
              f"- 风格轮动 `sector_rotation`：{best_in('sector_rotation')}",
              "",
              "**怎么读这份报告**：单边上涨里靠前的多是**满仓吃贝塔**的趋势/动量类，震荡市里靠前的往往是"
              "**均值回归与做市类**——这两类排名不必太当真，因为它们只是「性格对口」。"
              "真正值得进入组合的，是矩阵里横向 spread 小、worst-case 仍接近 0 甚至为正的那一批："
              "它们赚的是逻辑钱，不是行情钱。反过来，某个情景夏普很高而最差情景深负的，"
              "本质是**在赌行情性格**，需要先判断市场状态才敢用。",
              "",
              "**局限**：情景为人工合成、性格被刻意放大且互相独立，不存在真实市场的耦合与制度变化；"
              "同一批策略在同一批情景上排序，仍带有选择偏差与过拟合风险；成本为简化单边费率，未建模冲击成本。",
              "",
              "## 复现", "",
              "```bash",
              f"python examples/run_stress.py --kinds {','.join(kinds)} --n-days {meta['n_days']} "
              f"--n-assets {meta['n_assets']} --cost {meta['cost_rate']}",
              "```",
              "",
              f"产物：`stress_metrics.csv`（长表 {len(long_df)} 行：scenario/strategy/channel/sharpe/"
              "total_return/max_drawdown）与本报告。",
              ""]
    return "\n".join(lines)


def write_outputs(out_dir: str, long_df: pd.DataFrame, robust: pd.DataFrame,
                  sharpe: pd.DataFrame, bench: Dict[str, Dict[str, float]],
                  failed: Dict[str, List[str]], kinds: Sequence[str], meta: Dict) -> Tuple[str, str]:
    """落盘 CSV 长表与 Markdown 汇总，返回两个文件路径。"""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "stress_metrics.csv")
    md_path = os.path.join(out_dir, "STRESS_SUMMARY.md")
    long_df[METRIC_COLS].round(6).to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_summary_md(long_df, robust, sharpe, bench, failed, kinds, meta))
    return csv_path, md_path


def parse_kinds(raw: str) -> List[str]:
    """解析 --kinds（逗号分隔），校验并保持 SCENARIOS 的原始顺序。"""
    asked = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [k for k in asked if k not in SCENARIOS]
    if unknown:
        raise SystemExit(f"未知情景 {unknown}，可选：{', '.join(SCENARIOS)}")
    return [k for k in SCENARIOS if k in asked] or list(SCENARIOS)


def main() -> None:
    ap = argparse.ArgumentParser(description="策略压力/情景稳健性测试（合成数据、离线）")
    ap.add_argument("--kinds", default=",".join(SCENARIOS),
                    help=f"逗号分隔的情景列表，默认全部（可选：{','.join(SCENARIOS)}）")
    ap.add_argument("--cost", type=float, default=0.001, help="单边成本率，默认 0.001")
    ap.add_argument("--n-days", type=int, default=1000, help="每个情景的交易日数，默认 1000")
    ap.add_argument("--n-assets", type=int, default=8, help="每个情景的资产数，默认 8")
    ap.add_argument("--seed", type=int, default=2026, help="情景随机种子，默认 2026")
    ap.add_argument("--out", default=os.path.join(HERE, "research", "stress"),
                    help="产物目录，默认 research/stress")
    ap.add_argument("--top", type=int, default=10, help="控制台打印的稳健性排行条数，默认 10")
    a = ap.parse_args()

    kinds = parse_kinds(a.kinds)
    strategies = sorted(ks.discover(), key=lambda s: s.name)
    print("=" * 72)
    print(f"压力/情景稳健性测试：{len(strategies)} 策略 × {len(kinds)} 情景 "
          f"（{a.n_assets} 资产, {a.n_days} 期, 单边成本 {a.cost:.3%}, seed {a.seed}）")
    for k in kinds:
        print(f"  - {k:<16} {describe_scenario(k)}")
    print("-" * 72)

    t0 = time.time()
    long_df, bench, failed = run_stress(strategies, kinds, a.n_days, a.n_assets, a.seed, a.cost)
    if long_df.empty:
        raise SystemExit("所有策略在所有情景上都失败了，无可汇总结果。")
    robust, sharpe = robustness_table(long_df, kinds)
    meta = {"n_strategies": len(strategies), "n_ok": len(long_df), "n_days": a.n_days,
            "n_assets": a.n_assets, "seed": a.seed, "cost_rate": a.cost}
    csv_path, md_path = write_outputs(a.out, long_df, robust, sharpe, bench, failed, kinds, meta)

    labels = classify(robust, kinds, bench)
    print("-" * 72)
    print(f"完成 {len(long_df)} 条回测（失败 {sum(len(v) for v in failed.values())}），耗时 {time.time() - t0:.1f}s")
    print(f"稳健性 Top{a.top}（按最差情景夏普 worst-case）：")
    for i, (name, row) in enumerate(robust.head(a.top).iterrows(), 1):
        print(f"  {i:>2}. {name:<26} ch={row['channel']:<14} worst={row['worst_sharpe']:+.2f}"
              f"({row['worst_scenario']})  mean={row['mean_sharpe']:+.2f}  "
              f"正收益 {int(row['n_positive'])}/{int(row['n_scenarios'])}")
    print(f"分类：全天候 {len(labels['all_weather'])} ｜ 抗压 {len(labels['stress_resilient'])} "
          f"｜ 单边依赖 {len(labels['trend_dependent'])} ｜ 脆弱 {len(labels['fragile'])} "
          f"｜ 混合 {len(labels['mixed'])}")
    print(f"长表 -> {csv_path}")
    print(f"报告 -> {md_path}")


if __name__ == "__main__":
    main()
