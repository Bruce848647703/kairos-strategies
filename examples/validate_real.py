"""在真实 A 股数据上做样本外验证（walk-forward OOS / 参数敏感性 / PSR / bootstrap CI）。

运行：
  python examples/validate_real.py                 # quick 模式
  python examples/validate_real.py --full --data-dir /path/to/csv
产物： research/real/validation/<策略>/ 与 research/real/VALIDATION_SUMMARY.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import realdata, validation

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "ashare"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--cost", type=float, default=0.001)
    a = ap.parse_args()

    if not (os.path.isdir(a.data_dir) and any(f.endswith(".csv") for f in os.listdir(a.data_dir))):
        print(f"本地无数据，联网抓取 -> {a.data_dir}")
        realdata.fetch_universe(list(realdata.UNIVERSE), a.data_dir, start="2016-01-01")

    data = realdata.load_panel(a.data_dir)
    strategies = ks.discover()
    mode = "full" if a.full else "quick"
    print(f"真实数据验证：{data.prices.shape[0]} 日 × {data.n_assets} 只，"
          f"{len(strategies)} 策略，模式 {mode}")

    root = os.path.join(HERE, "research", "real")
    rows = validation.run_validation(data, strategies, research_root=root,
                                     mode=mode, cost_rate=a.cost)
    print("=" * 60)
    print(f"完成 -> {os.path.join(root, 'VALIDATION_SUMMARY.md')}")
    top = sorted(rows, key=lambda r: r.get("psr", 0), reverse=True)[:8]
    print("真实数据 PSR Top8：")
    for r in top:
        print(f"  {r['name']:<24} ch={r['channel']:<13} psr={r.get('psr',0):.3f} "
              f"sharpe={r.get('sharpe',0):.2f} pos_win={(r.get('positive_window_ratio') or 0)*100:.0f}%")


if __name__ == "__main__":
    main()
