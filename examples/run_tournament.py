"""策略锦标赛：在真实(或合成)数据上对全部策略做相关性聚类 + 去相关精选 + 分散化组合。

运行：
  python examples/run_tournament.py --data-dir <ashare_csv_dir>     # 真实 A 股
  python examples/run_tournament.py --synthetic                     # 合成数据(离线快速)
产物： research/tournament/{REPORT.md, ranking.csv, correlation.csv, selection.json, combo_equity.*, strategy_returns.csv}
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import realdata, tournament

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "ashare"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--corr-thresh", type=float, default=0.7)
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--combo", default="inverse_vol", choices=["inverse_vol", "equal", "sharpe"])
    ap.add_argument("--cost", type=float, default=0.001)
    a = ap.parse_args()

    if a.synthetic:
        data = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    else:
        if not (os.path.isdir(a.data_dir) and any(f.endswith(".csv") for f in os.listdir(a.data_dir))):
            print(f"本地无数据，联网抓取 -> {a.data_dir}")
            realdata.fetch_universe(list(realdata.UNIVERSE), a.data_dir, start="2016-01-01")
        data = realdata.load_panel(a.data_dir)

    strategies = ks.discover()
    print(f"锦标赛：{len(strategies)} 策略 × 数据 {data.name}"
          f"（{data.n_assets} 资产, {len(data.dates)} 期）")
    out = os.path.join(HERE, "research", "tournament")
    res = tournament.run_tournament(
        data, strategies, out, top_k=a.top_k, corr_thresh=a.corr_thresh,
        n_clusters=a.clusters, combo_method=a.combo, cost_rate=a.cost)

    cm = res["combo_metrics"]
    print("=" * 60)
    print(f"去相关精选({len(res['selected'])}): {', '.join(res['selected'])}")
    print(f"精英组合({a.combo}): 夏普 {cm['sharpe']:.2f} ｜ 累计 {cm['total_return']*100:.1f}% "
          f"｜ 回撤 {cm['max_drawdown']*100:.1f}% ｜ 卡玛 {cm['calmar']:.2f}")
    print(f"聚类簇数 {len(res['clusters'])}；失败 {len(res['failed'])}")
    print(f"报告 -> {os.path.join(out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
