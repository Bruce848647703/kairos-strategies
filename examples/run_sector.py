"""在真实 A 股数据上，对全部策略叠加「行业均衡」后处理再回测。

行业均衡(sector_equalize)：保持只做多与每期总敞口不变，把权重在当期实际持有的行业间均衡，
消除行业集中度（用 realdata.SECTOR_MAP 的 8 个行业分组）。

运行： python examples/run_sector.py [--data-dir /path/to/csv] [--cost 0.001]
产物： research/real_sector/records/<策略>/ 与 research/real_sector/SUMMARY.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import realdata, report, sector
from kairos_strategies.engine import Backtester

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Var:
    """给 record_readme 提供带「行业均衡」标注的 meta。"""

    def __init__(self, s):
        self._s = s
        self.name = s.name
        self.channel = s.channel

    def meta(self):
        m = dict(self._s.meta())
        m["description"] = "[行业均衡] " + (m.get("description") or "")
        m["universe"] = m.get("universe", "") + "+sector_equalize"
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "ashare"))
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--no-chart", action="store_true")
    a = ap.parse_args()

    if not (os.path.isdir(a.data_dir) and any(f.endswith(".csv") for f in os.listdir(a.data_dir))):
        print(f"本地无数据，联网抓取 -> {a.data_dir}")
        realdata.fetch_universe(list(realdata.UNIVERSE), a.data_dir, start="2016-01-01")

    data = realdata.load_panel(a.data_dir)
    strategies = ks.discover()
    bt = Backtester(cost_rate=a.cost, periods_per_year=data.periods_per_year)
    root = os.path.join(HERE, "research", "real_sector")
    records = os.path.join(root, "records")
    os.makedirs(records, exist_ok=True)
    print(f"真实数据×行业均衡：{data.prices.shape[0]} 日 × {data.n_assets} 只，{len(strategies)} 策略")

    rows, failed = [], []
    for s in strategies:
        try:
            w = sector.sector_equalize(s.generate_weights(data), realdata.SECTOR_MAP)
            res = bt.run(data, w)
            m = dict(res._metrics)
            d = os.path.join(records, s.name)
            os.makedirs(d, exist_ok=True)
            chart = (not a.no_chart) and report.save_chart(
                res.equity, os.path.join(d, "equity.png"), f"{s.name} (sector-equalized) equity")
            res.equity.to_frame("equity").to_csv(os.path.join(d, "equity.csv"))
            with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": _Var(s).meta(), "metrics": m, "variant": "sector_equalize",
                           "data": data.name, "cost_rate": a.cost}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
                f.write(report.record_readme(_Var(s), m, data, bt, chart))
            rows.append({"name": s.name, "channel": s.channel, **m})
        except Exception as e:  # noqa: BLE001
            failed.append((s.name, s.channel, repr(e)))
            print(f"  [skip] {s.name}: {e}")

    with open(os.path.join(root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(report.summary_md(rows, data, bt))
    notes = [
        "# 行业均衡回测说明 (SECTOR-EQUALIZED, REAL DATA)",
        "",
        f"- 数据：真实 A 股 {data.n_assets} 只，{data.prices.shape[0]} 交易日，"
        f"{data.dates[0].date()} ~ {data.dates[-1].date()}。",
        f"- 处理：对每个策略的原始权重做 `sector_equalize`（8 行业分组见 realdata.SECTOR_MAP），"
        f"在当期实际持有行业间均衡、保持总敞口与只做多属性，消除行业集中度。",
        f"- 成本：单边 {a.cost:.4%}；成功 {len(rows)} / 跳过 {len(failed)}。",
        "",
        "**重要**：真实历史数据演示，存在过拟合/幸存者偏差等局限，不构成投资建议。",
        "",
    ]
    if failed:
        notes += ["## 跳过", ""] + [f"- `{n}` ({c}): {e}" for n, c, e in failed]
    with open(os.path.join(root, "NOTES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")

    print("=" * 60)
    print(f"完成：成功 {len(rows)} / 跳过 {len(failed)} -> {root}")
    top = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:8]
    print("行业均衡后 夏普 Top8：")
    for r in top:
        print(f"  {r['name']:<24} ch={r['channel']:<13} sharpe={r['sharpe']:.2f} "
              f"ret={r['total_return']*100:6.1f}% mdd={r['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
