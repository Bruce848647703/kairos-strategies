"""在真实多资产 ETF 数据上跑「资产配置类」策略，产出研究记录。

universe = realdata.ETF_UNIVERSE（A股/海外股/黄金/国债/货币 9 只 ETF）。
策略 = multi_asset(全天候/GTAA/60-40) + 选定的 allocation/taa/trend/benchmark 渠道策略。

运行：
  python examples/run_multi.py --data-dir <etf_csv_dir>
  python examples/run_multi.py            # 无本地数据则联网抓取
产物： research/real_multi/records/<策略>/ 与 SUMMARY.md、NOTES.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import multi_asset, realdata, report
from kairos_strategies.engine import Backtester

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITELIST = {"allocation", "taa", "trend", "benchmark"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "etf"))
    ap.add_argument("--cost", type=float, default=0.0005, help="ETF 单边成本(默认万五)")
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()

    if not (os.path.isdir(a.data_dir) and any(f.endswith(".csv") for f in os.listdir(a.data_dir))):
        print(f"本地无 ETF 数据，联网抓取 -> {a.data_dir}")
        realdata.fetch_universe(list(realdata.ETF_UNIVERSE), a.data_dir, start="2012-01-01")

    data = realdata.load_panel(a.data_dir)
    data.name = "ashare_etf_real"
    strategies = [c() for c in multi_asset.MULTI_ASSET_STRATEGIES]
    strategies += [s for s in ks.discover() if s.channel in WHITELIST]
    bt = Backtester(cost_rate=a.cost, periods_per_year=data.periods_per_year)
    root = os.path.join(HERE, "research", "real_multi")
    records = os.path.join(root, "records")
    os.makedirs(records, exist_ok=True)
    print(f"真实多资产：{data.prices.shape[0]} 交易日 × {data.n_assets} 只 ETF，"
          f"{data.dates[0].date()}~{data.dates[-1].date()}；{len(strategies)} 个配置类策略")

    rows, failed = [], []
    for s in strategies:
        try:
            rows.append(report.run_strategy(s, data, bt, records, chart=not a.no_chart))
        except Exception as e:  # noqa: BLE001
            failed.append((s.name, getattr(s, "channel", "?"), repr(e)))
            print(f"  [skip] {s.name}: {e}")

    with open(os.path.join(root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(report.summary_md(rows, data, bt))
    notes = [
        "# 真实多资产配置回测说明 (REAL MULTI-ASSET)",
        "",
        f"- 数据：真实跨资产 ETF（A股宽基/成长、海外股、黄金、国债、货币），{data.n_assets} 只，"
        f"{data.prices.shape[0]} 交易日，{data.dates[0].date()} ~ {data.dates[-1].date()}。",
        f"- 资产类别：{ {c: m for c, m in realdata.ASSET_CLASSES.items()} }",
        f"- 策略：multi_asset(全天候/GTAA/60-40) + allocation/taa/trend/benchmark 渠道。",
        f"- 成本：单边 {a.cost:.4%}；成功 {len(rows)} / 跳过 {len(failed)}。",
        "",
        "**重要**：真实历史数据演示，存在样本区间依赖等局限，不构成投资建议。",
        "",
    ]
    if failed:
        notes += ["## 跳过", ""] + [f"- `{n}` ({c}): {e}" for n, c, e in failed]
    with open(os.path.join(root, "NOTES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")

    print("=" * 60)
    print(f"完成：成功 {len(rows)} / 跳过 {len(failed)} -> {root}")
    for r in sorted(rows, key=lambda x: x["sharpe"], reverse=True):
        print(f"  {r['name']:<22} ch={r['channel']:<12} sharpe={r['sharpe']:.2f} "
              f"ret={r['total_return']*100:7.1f}% mdd={r['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
