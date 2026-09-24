"""在真实 A 股数据上跑全部策略，产出真实回测研究记录与汇总。

运行：
  python examples/run_real.py                 # 用本地 data/ashare（无则自动联网抓取）
  python examples/run_real.py --data-dir /path/to/csv --cost 0.001
  python examples/run_real.py --fetch          # 强制重新联网抓取

产物： research/real/records/<策略>/ 与 research/real/SUMMARY.md、REAL_NOTES.md
说明： 真实数据来自公开行情接口，仅用于研究演示，结果不构成投资建议。
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import realdata, report
from kairos_strategies.engine import Backtester

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "ashare"))
    ap.add_argument("--cost", type=float, default=0.001, help="单边成本率(默认千一)")
    ap.add_argument("--fetch", action="store_true", help="强制联网重新抓取")
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()

    have = os.path.isdir(a.data_dir) and any(f.endswith(".csv") for f in os.listdir(a.data_dir))
    if a.fetch or not have:
        print(f"联网抓取真实 A 股日线 -> {a.data_dir} ...")
        res = realdata.fetch_universe(list(realdata.UNIVERSE), a.data_dir, start="2016-01-01")
        print(f"  抓取完成，{sum(1 for v in res.values() if v>=200)}/{len(res)} 只有效")

    data = realdata.load_panel(a.data_dir)
    print(f"真实数据: {data.prices.shape[0]} 交易日 × {data.n_assets} 只 A 股，"
          f"区间 {data.dates[0].date()} ~ {data.dates[-1].date()}")

    strategies = ks.discover()
    bt = Backtester(cost_rate=a.cost, periods_per_year=data.periods_per_year)
    root = os.path.join(HERE, "research", "real")
    records_root = os.path.join(root, "records")
    os.makedirs(records_root, exist_ok=True)

    rows, failed = [], []
    for s in strategies:
        try:
            rows.append(report.run_strategy(s, data, bt, records_root, chart=not a.no_chart))
        except Exception as e:  # noqa: BLE001
            failed.append((s.name, s.channel, repr(e)))
            print(f"  [skip] {s.name} ({s.channel}): {e}")

    with open(os.path.join(root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(report.summary_md(rows, data, bt))

    notes = [
        "# 真实数据回测说明 (REAL NOTES)",
        "",
        f"- 数据：真实 A 股前复权日线，来源为公开行情接口（腾讯/新浪），见 kairos-data 的 DATA_NOTICE。",
        f"- 规模：{data.n_assets} 只跨行业流动股，{data.prices.shape[0]} 个交易日，"
        f"{data.dates[0].date()} ~ {data.dates[-1].date()}。",
        f"- 回测：向量化、权重滞后一期防未来、单边成本 {a.cost:.4%}；已处理停牌/上市日/非正价。",
        f"- 成功 {len(rows)} 个策略，跳过 {len(failed)} 个。",
        "",
        "**重要**：结果为历史真实数据上的演示，存在过拟合/幸存者偏差等局限，**不构成任何投资建议**。",
        "",
    ]
    if failed:
        notes += ["## 跳过的策略", ""] + [f"- `{n}` ({c}): {e}" for n, c, e in failed]
    with open(os.path.join(root, "REAL_NOTES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")

    print("=" * 60)
    print(f"真实数据回测完成：成功 {len(rows)} / 跳过 {len(failed)} -> {root}")
    top = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:8]
    print("\n真实数据夏普 Top8：")
    for r in top:
        print(f"  {r['name']:<24} ch={r['channel']:<13} sharpe={r['sharpe']:.2f} "
              f"ret={r['total_return']*100:6.1f}% mdd={r['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
