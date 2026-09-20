"""跑通完整 QR 流程：发现全部策略 -> 合成数据 -> 回测 -> 生成研究记录与汇总。

运行： python examples/run_all.py
产物： research/records/<策略>/ 与 research/SUMMARY.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks


def main():
    data = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    strategies = ks.discover()
    grouped = ks.by_channel(strategies)

    print("=" * 64)
    print(f"发现策略 {len(strategies)} 个，覆盖 {len(grouped)} 个渠道：")
    for ch, items in grouped.items():
        print(f"  - {ch:<12} ({len(items)}): {', '.join(s.name for s in items)}")

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
    rows = ks.report.run_all(data, strategies, research_root=root, cost_rate=0.0005, chart=True)

    print("=" * 64)
    print(f"已生成 {len(rows)} 份研究记录 -> {os.path.join(root, 'records')}")
    print(f"汇总表 -> {os.path.join(root, 'SUMMARY.md')}")
    # 控制台打印夏普前五
    top = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:5]
    print("\n夏普 Top5：")
    for r in top:
        print(f"  {r['name']:<22} ch={r['channel']:<12} sharpe={r['sharpe']:.2f} "
              f"ret={r['total_return']*100:.1f}% mdd={r['max_drawdown']*100:.1f}%")


if __name__ == "__main__":
    main()
