"""跑通 QR 流程「评估」环节升级版：对全部策略做样本外验证 / 稳健性分析。

内容：walk-forward 样本外窗口 + 参数敏感性(OAT) + 概率夏普 PSR + bootstrap 夏普置信区间。

运行： python examples/validate.py          # quick 模式（默认，加速）
      python examples/validate.py --full   # full 模式（更多窗口/重采样）
产物： research/validation/<策略>/{report.md, validation.json}
      research/VALIDATION_SUMMARY.md（按 PSR 降序汇总）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_strategies as ks
from kairos_strategies import validation


def main():
    mode = "full" if "--full" in sys.argv[1:] else "quick"
    data = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    strategies = ks.discover()

    print("=" * 64)
    print(f"发现策略 {len(strategies)} 个，验证模式：{mode}")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
    t0 = time.time()
    rows = validation.run_validation(data, strategies, research_root=root, mode=mode)
    elapsed = time.time() - t0

    ok = [r for r in rows if r.get("error") is None]
    bad = [r for r in rows if r.get("error") is not None]
    print("=" * 64)
    print(f"验证完成：成功 {len(ok)} / 失败 {len(bad)}，耗时 {elapsed:.1f}s "
          f"（平均 {elapsed / max(len(rows), 1):.2f}s/策略）")
    print(f"逐策略产物 -> {os.path.join(root, 'validation')}")
    print(f"汇总表     -> {os.path.join(root, 'VALIDATION_SUMMARY.md')}")

    top = sorted(ok, key=lambda r: r["psr"], reverse=True)[:5]
    print("\nPSR Top5：")
    for r in top:
        print(f"  {r['name']:<24} ch={r['channel']:<12} psr={r['psr']:.3f} "
              f"sharpe={r['sharpe']:.2f} ci=[{r['sharpe_ci_lo']:.2f},{r['sharpe_ci_hi']:.2f}] "
              f"pos_win={r['positive_window_ratio'] * 100:.0f}% sens_range={r['sens_range']:.2f}")
    for r in bad:
        print(f"  [失败] {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
