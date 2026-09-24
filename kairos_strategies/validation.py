"""样本外验证 / 稳健性工具集（QR 流程「评估」环节升级）。

为已回测的策略提供 100% 原创的四类验证工具（纯 numpy/pandas；scipy 可选，
缺失时正态 CDF 自动退化为 math.erf 实现）：

1. `walk_forward_oos`       把时间轴切成 n 段连续样本外窗口，逐段用 Backtester
                            回测并汇总各窗口指标与跨窗口一致性统计；
2. `parameter_sensitivity`  对数值型参数做 OAT（一次一个）扰动，量化夏普对参数
                            的敏感度：稳健性区间 = max−min，越小越稳健；
3. `probabilistic_sharpe`   概率夏普比率 PSR（Bailey & López de Prado, 2014），
                            用收益的偏度/峰度校正样本夏普的统计显著性；
4. `bootstrap_sharpe_ci`    移动块 bootstrap（保留自相关）给年化夏普的置信区间，
                            固定 seed、确定性可复现。

`validate_strategy` 汇总以上四项并落盘 report.md / validation.json；
`run_validation` 批量验证多个策略并生成 VALIDATION_SUMMARY.md（按 PSR 降序）。

约定：离线、确定性、Python 3.9 兼容；本模块只读取既有组件，不修改任何已有文件。
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import MarketData, Strategy
from .engine import Backtester

try:
    from scipy.stats import norm as _scipy_norm
except Exception:
    _scipy_norm = None

__all__ = [
    "walk_forward_oos",
    "parameter_sensitivity",
    "probabilistic_sharpe",
    "bootstrap_sharpe_ci",
    "validate_strategy",
    "run_validation",
]

_SQRT_2 = math.sqrt(2.0)

QUICK_CFG: Dict[str, int] = {"n_windows": 3, "n_boot": 200, "max_evals": 24}
FULL_CFG: Dict[str, int] = {"n_windows": 4, "n_boot": 500, "max_evals": 60}


def _norm_cdf(x: float) -> float:
    """标准正态 CDF Φ(x)：优先 scipy，缺失时用 math.erf 精确实现。"""
    if _scipy_norm is not None:
        try:
            return float(_scipy_norm.cdf(x))
        except Exception:
            pass
    return 0.5 * (1.0 + math.erf(float(x) / _SQRT_2))


def _round_half_up(x: float) -> int:
    """四舍五入取整（内置 round 是银行家舍入，不符合常规「四舍五入」语义）。"""
    return int(math.floor(float(x) + 0.5))


def _is_numeric_param(v: Any) -> bool:
    """判断参数是否为可扰动的数值型（排除 bool）。"""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _fmt(v: Any, pct: bool = False, nd: int = 4) -> str:
    """报告用数值格式化：None→NA，inf/nan 有可读写法。"""
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f * 100:.2f}%" if pct else f"{f:.{nd}f}"


def _jsonable(obj: Any) -> Any:
    """递归转成可 JSON 序列化结构：NaN/inf→None，numpy 标量→python 标量。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _as_clean_returns(returns: Any) -> pd.Series:
    """把任意收益输入（Series/ndarray/list）清洗为有限值 Series。"""
    r = pd.Series(np.asarray(returns, dtype="float64").ravel()).dropna()
    return r[np.isfinite(r)]


def walk_forward_oos(data: MarketData, strategy: Strategy, n_windows: int = 4,
                     cost_rate: float = 0.001) -> Dict:
    """Walk-forward 样本外验证：把时间轴切成 n_windows 段连续窗口逐段回测。

    做法：策略 `generate_weights` 用全量 data 生成一次（策略内部已防未来函数，
    引擎再滞后一期，双重保险），随后按窗口切片价格与权重，对每段单独用
    `Backtester` 回测。注意每个窗口首期因引擎滞后无收益、且计一次建仓成本，
    结果略偏保守，但对跨窗口一致性比较无实质影响。

    返回 dict：
      - windows: 每窗口 {window,start,end,n_periods,total_return,sharpe,...}
      - positive_window_ratio: 正收益窗口占比 ∈ [0,1]
      - window_sharpe_mean / window_sharpe_std: 窗口夏普的均值/标准差(ddof=1)
      - best_window_return / worst_window_return
    """
    n_windows = max(1, int(n_windows))
    dates = data.dates
    n = len(dates)
    weights = strategy.generate_weights(data).reindex(index=dates, columns=data.symbols)
    bt = Backtester(cost_rate=cost_rate, risk_free=0.0,
                    periods_per_year=data.periods_per_year)
    windows: List[Dict] = []
    for k, chunk in enumerate(np.array_split(np.arange(n), n_windows)):
        if len(chunk) == 0:
            continue
        i0, i1 = int(chunk[0]), int(chunk[-1]) + 1
        sub = MarketData(
            prices=data.prices.iloc[i0:i1],
            volumes=None if data.volumes is None else data.volumes.iloc[i0:i1],
            periods_per_year=data.periods_per_year,
            name=f"{data.name}_w{k + 1}")
        m = bt.run(sub, weights.iloc[i0:i1]).metrics
        windows.append({
            "window": k + 1,
            "start": str(dates[i0].date()),
            "end": str(dates[i1 - 1].date()),
            "n_periods": int(i1 - i0),
            "total_return": float(m["total_return"]),
            "sharpe": float(m["sharpe"]),
            "volatility": float(m["volatility"]),
            "max_drawdown": float(m["max_drawdown"]),
        })
    rets = [w["total_return"] for w in windows]
    shps = [w["sharpe"] for w in windows]
    return {
        "n_windows": len(windows),
        "windows": windows,
        "positive_window_ratio": float(np.mean([1.0 if x > 0 else 0.0 for x in rets])) if rets else 0.0,
        "window_sharpe_mean": float(np.mean(shps)) if shps else 0.0,
        "window_sharpe_std": float(np.std(shps, ddof=1)) if len(shps) > 1 else 0.0,
        "best_window_return": float(np.max(rets)) if rets else 0.0,
        "worst_window_return": float(np.min(rets)) if rets else 0.0,
    }


def parameter_sensitivity(data: MarketData, strategy: Strategy,
                          perturb: Sequence[float] = (0.5, 1.0, 2.0),
                          cost_rate: float = 0.001, max_evals: int = 60) -> Dict:
    """参数敏感性分析（OAT，一次只扰动一个参数）。

    对 `strategy.params` 中每个数值型参数（排除 bool），按 perturb 系数逐一
    缩放（int 参数四舍五入且钳制 ≥1），每个组合用 `copy.deepcopy(strategy)`
    构造扰动实例并把 params 复制为实例级 dict 后修改——既不污染原实例，也不
    污染类属性。扰动值与 ×1.0 组合等价于原参数时直接复用基线夏普，不重跑。
    若某扰动点回测抛异常或夏普非有限值则跳过该点。`max_evals` 对额外回测
    次数做上限保护，超限即截断（结果中 capped=True）。

    返回 dict：
      - baseline_sharpe: 原参数夏普
      - params: {参数名: {original, points, value_sharpe({取值:夏普}), min, median, max}}
      - overall: 全部评估点夏普的 {min, median, max, range}，range=max−min 即
        「稳健性区间」，越小代表策略对参数越不敏感、越稳健
      - n_numeric_params / n_evals / n_skipped / capped
    """
    bt = Backtester(cost_rate=cost_rate, risk_free=0.0,
                    periods_per_year=data.periods_per_year)
    base_sharpe = float(bt.run(data, strategy.generate_weights(data)).metrics["sharpe"])
    params = dict(strategy.params)
    numeric = {k: v for k, v in params.items() if _is_numeric_param(v)}

    results: Dict[str, Dict] = {}
    all_sharpes: List[float] = [base_sharpe]
    n_evals = 0
    n_skipped = 0
    capped = False
    for key, orig in numeric.items():
        if capped:
            break
        is_int = isinstance(orig, (int, np.integer))
        orig_val: Any = int(orig) if is_int else float(orig)
        points: List[Dict] = []
        seen = set()
        has_orig_point = False
        for f in perturb:
            f = float(f)
            new_val: Any = max(1, _round_half_up(orig_val * f)) if is_int else orig_val * f
            if new_val in seen:
                continue
            seen.add(new_val)
            if new_val == orig_val:
                points.append({"value": new_val, "factor": f,
                               "sharpe": base_sharpe, "reused_baseline": True})
                has_orig_point = True
                continue
            if n_evals >= int(max_evals):
                capped = True
                break
            n_evals += 1
            trial = copy.deepcopy(strategy)
            trial.params = dict(params)
            trial.params[key] = new_val
            try:
                sh = float(bt.run(data, trial.generate_weights(data)).metrics["sharpe"])
            except Exception:
                n_skipped += 1
                continue
            if not math.isfinite(sh):
                n_skipped += 1
                continue
            points.append({"value": new_val, "factor": f, "sharpe": sh})
            all_sharpes.append(sh)
        if not has_orig_point:
            points.append({"value": orig_val, "factor": 1.0,
                           "sharpe": base_sharpe, "reused_baseline": True})
        points.sort(key=lambda p: p["value"])
        sh_list = [p["sharpe"] for p in points]
        results[key] = {
            "original": orig_val,
            "points": points,
            "value_sharpe": {str(p["value"]): p["sharpe"] for p in points},
            "min": float(np.min(sh_list)),
            "median": float(np.median(sh_list)),
            "max": float(np.max(sh_list)),
        }
    return {
        "baseline_sharpe": base_sharpe,
        "params": results,
        "overall": {
            "min": float(np.min(all_sharpes)),
            "median": float(np.median(all_sharpes)),
            "max": float(np.max(all_sharpes)),
            "range": float(np.max(all_sharpes) - np.min(all_sharpes)),
        },
        "n_numeric_params": len(numeric),
        "n_evals": n_evals,
        "n_skipped": n_skipped,
        "capped": capped,
    }


def _psr_components(returns: Any, sr_benchmark: float = 0.0) -> Dict[str, Any]:
    """PSR 的中间量与结果（供 probabilistic_sharpe 与报告复用）。

    公式（Bailey & López de Prado, 2014）：
        PSR = Φ( (SR−SR*)·√(n−1) / √(1 − skew·SR + ((kurt−1)/4)·SR²) )
    其中 SR 为每期（非年化）样本夏普 = mean/std(ddof=1)；skew 为样本偏度、
    kurt 为样本超额峰度（pandas 约定）；SR* 为基准夏普（同为每期口径）；
    Φ 为标准正态 CDF。样本不足（n<3）返回中性值 0.5；零方差时按均值是否
    超过基准返回 1.0/0.5。
    """
    r = _as_clean_returns(returns)
    n = int(len(r))
    comp: Dict[str, Any] = {"n": n, "sr_benchmark": float(sr_benchmark),
                            "sr": None, "skew": None, "kurt": None, "psr": 0.5}
    if n < 3:
        return comp
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd < 1e-15:
        comp["psr"] = 1.0 if mu > float(sr_benchmark) else 0.5
        return comp
    sr = mu / sd
    skew = float(r.skew())
    kurt = float(r.kurt())
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 0.0
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    num = (sr - float(sr_benchmark)) * math.sqrt(n - 1)
    if denom <= 1e-12:
        psr = 1.0 if num > 0 else (0.5 if num == 0.0 else 0.0)
    else:
        psr = _norm_cdf(num / math.sqrt(denom))
    comp.update({"sr": sr, "skew": skew, "kurt": kurt,
                 "psr": float(min(1.0, max(0.0, psr)))})
    return comp


def probabilistic_sharpe(returns: Any, sr_benchmark: float = 0.0) -> float:
    """概率夏普比率 PSR：样本夏普显著大于基准 SR* 的概率（偏度/峰度校正）。

    输入每期收益序列（Series/ndarray/list），SR 与 sr_benchmark 均为每期
    （非年化）口径。返回 [0,1] 之间的概率；接近 1 表示夏普统计上显著。
    """
    return _psr_components(returns, sr_benchmark)["psr"]


def bootstrap_sharpe_ci(returns: Any, n_boot: int = 500, ci: float = 0.95,
                        block: int = 20, seed: int = 0,
                        periods_per_year: int = 252) -> Tuple[float, float]:
    """移动块 bootstrap 给「年化夏普」的置信区间（保留收益自相关结构）。

    每次重采样抽取 ceil(n/block) 个长度为 block 的连续块（起点均匀随机），
    拼接并截断到 n，计算年化夏普；取百分位区间 [(1−ci)/2, 1−(1−ci)/2]。
    固定 seed 完全确定性；block 自动钳制到 [1, n]。返回 (lo, hi)。
    """
    arr = _as_clean_returns(returns).values
    n = int(arr.shape[0])
    if n < 2:
        return (0.0, 0.0)
    n_boot = max(1, int(n_boot))
    block = int(max(1, min(int(block), n)))
    ci = float(min(max(ci, 0.0), 1.0))
    rng = np.random.default_rng(int(seed))
    m = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, m))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, m * block)[:, :n]
    samples = arr[idx]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    ann = math.sqrt(periods_per_year)
    sh = np.where(sd > 1e-15, mu / np.where(sd > 1e-15, sd, 1.0) * ann, 0.0)
    q_lo = (1.0 - ci) / 2.0 * 100.0
    lo, hi = np.percentile(sh, [q_lo, 100.0 - q_lo])
    return (float(lo), float(hi))


def _validation_md(sm: Dict) -> str:
    """把 validate_strategy 的汇总 dict 渲染为中文 Markdown 报告。"""
    b = sm["baseline"]
    wf = sm["walk_forward"]
    sens = sm["sensitivity"]
    ci = sm["sharpe_ci"]
    d = sm["data"]
    psr_d = sm["psr_detail"]
    ov = sens["overall"]
    L: List[str] = []
    L.append(f"# 策略验证报告：{sm['name']}")
    L.append("")
    L.append(f"> 渠道：**{sm['channel']}** ｜ 模式：{sm['mode']} ｜ 单边成本：{sm['cost_rate']:.4%}  ")
    L.append(f"> 数据：{d['name']}（{d['n_assets']} 资产，{d['n_periods']} 期，{d['start']} ~ {d['end']}）  ")
    ptxt = "、".join(f"`{k}`={v}" for k, v in sm["params"].items()) or "（无）"
    L.append(f"> 参数：{ptxt}")
    L.append("")
    L.append("## 1. 基线回测（全样本参考）")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 累计收益 | {_fmt(b['total_return'], pct=True)} |")
    L.append(f"| 年化收益 (CAGR) | {_fmt(b['cagr'], pct=True)} |")
    L.append(f"| 年化波动 | {_fmt(b['volatility'], pct=True)} |")
    L.append(f"| 夏普 | {_fmt(b['sharpe'], nd=2)} |")
    L.append(f"| 索提诺 | {_fmt(b['sortino'], nd=2)} |")
    L.append(f"| 最大回撤 | {_fmt(b['max_drawdown'], pct=True)} |")
    L.append(f"| 胜率 | {_fmt(b['win_rate'], pct=True)} |")
    L.append(f"| 累计换手 | {_fmt(b['total_turnover'], nd=2)} |")
    L.append("")
    L.append(f"## 2. Walk-Forward 样本外验证（{wf['n_windows']} 窗口）")
    L.append("")
    L.append("权重用全量数据一次性生成（策略内部已防未来函数），再按连续窗口切片、"
             "逐窗口独立回测。每窗口首期因引擎滞后无收益并计一次建仓成本，口径略保守。")
    L.append("")
    L.append("| 窗口 | 区间 | 期数 | 累计收益 | 年化波动 | 最大回撤 | 夏普 |")
    L.append("|---|---|---|---|---|---|---|")
    for w in wf["windows"]:
        L.append(f"| {w['window']} | {w['start']} ~ {w['end']} | {w['n_periods']} "
                 f"| {_fmt(w['total_return'], pct=True)} | {_fmt(w['volatility'], pct=True)} "
                 f"| {_fmt(w['max_drawdown'], pct=True)} | {_fmt(w['sharpe'], nd=2)} |")
    L.append("")
    L.append(f"- 正收益窗口占比：**{_fmt(wf['positive_window_ratio'], pct=True)}**")
    L.append(f"- 窗口夏普：均值 {_fmt(wf['window_sharpe_mean'], nd=2)}，"
             f"标准差 {_fmt(wf['window_sharpe_std'], nd=2)}")
    L.append(f"- 最佳/最差窗口累计收益：{_fmt(wf['best_window_return'], pct=True)} / "
             f"{_fmt(wf['worst_window_return'], pct=True)}")
    L.append("")
    L.append("## 3. 参数敏感性（OAT 一次一个扰动）")
    L.append("")
    tail = "，已达评估上限被截断。" if sens["capped"] else "。"
    L.append(f"对每个数值型参数按 ×0.5 / ×1.0 / ×2.0 逐一扰动（×1.0 复用基线；int 参数"
             f"四舍五入且 ≥1；异常或 NaN 的扰动点跳过）。基线夏普 "
             f"{_fmt(sens['baseline_sharpe'], nd=2)}，额外回测 {sens['n_evals']} 次，"
             f"跳过 {sens['n_skipped']} 个点{tail}")
    L.append("")
    if sens["params"]:
        L.append("| 参数 | 原值 | 扰动点（取值 → 夏普） | min | median | max |")
        L.append("|---|---|---|---|---|---|")
        for k, info in sens["params"].items():
            pts = "，".join(f"{p['value']} → {_fmt(p['sharpe'], nd=2)}" for p in info["points"])
            L.append(f"| `{k}` | {info['original']} | {pts} | {_fmt(info['min'], nd=2)} "
                     f"| {_fmt(info['median'], nd=2)} | {_fmt(info['max'], nd=2)} |")
        L.append("")
        L.append(f"- 全体评估点夏普：min {_fmt(ov['min'], nd=2)} / median {_fmt(ov['median'], nd=2)} "
                 f"/ max {_fmt(ov['max'], nd=2)}")
        L.append(f"- **稳健性区间（max−min）= {_fmt(ov['range'], nd=2)}**，越小说明对参数越不敏感、越稳健。")
    else:
        L.append("（该策略无数值型参数，OAT 扰动不适用。）")
    L.append("")
    L.append("## 4. 概率夏普比率（PSR）")
    L.append("")
    L.append("PSR = Φ((SR−SR*)·√(n−1) / √(1 − skew·SR + ((kurt−1)/4)·SR²))；SR 为每期"
             "（非年化）样本夏普，skew/kurt 为收益偏度与超额峰度，基准 SR*=0，Φ 为标准正态 CDF。")
    L.append("")
    L.append("| 输入 | 值 |")
    L.append("|---|---|")
    L.append(f"| 样本期数 n | {psr_d['n']} |")
    L.append(f"| 每期 SR | {_fmt(psr_d['sr'])} |")
    L.append(f"| 偏度 skew | {_fmt(psr_d['skew'])} |")
    L.append(f"| 超额峰度 kurt | {_fmt(psr_d['kurt'])} |")
    L.append(f"| **PSR** | **{_fmt(sm['psr'], nd=4)}** |")
    L.append("")
    L.append("## 5. Bootstrap 夏普置信区间（移动块法）")
    L.append("")
    L.append(f"移动块 bootstrap（block={ci['block']}，保留收益自相关），重采样 {ci['n_boot']} 次"
             f"（seed={ci['seed']}，确定性可复现），取年化夏普的 {ci['ci'] * 100:.0f}% 百分位区间。")
    L.append("")
    L.append(f"- 年化夏普点估计：**{_fmt(ci['point_annualized_sharpe'], nd=2)}**")
    L.append(f"- {ci['ci'] * 100:.0f}% 置信区间：**[{_fmt(ci['lo'], nd=2)}, {_fmt(ci['hi'], nd=2)}]**")
    L.append("- 区间下界 > 0 时，可认为夏普在该置信水平下显著为正。")
    L.append("")
    L.append("## 6. 结论")
    L.append("")
    if sm["psr"] >= 0.95:
        L.append(f"- PSR = {_fmt(sm['psr'], nd=3)} ≥ 0.95：样本夏普在统计上显著（偏度/峰度校正后）。")
    elif sm["psr"] >= 0.85:
        L.append(f"- PSR = {_fmt(sm['psr'], nd=3)}：有一定显著性证据，但未达 0.95 的常用阈值。")
    else:
        L.append(f"- PSR = {_fmt(sm['psr'], nd=3)} < 0.85：夏普显著高于 0 的证据不足。")
    L.append(f"- 样本外正收益窗口占比 {_fmt(wf['positive_window_ratio'], pct=True)}"
             f"（共 {wf['n_windows']} 窗），窗口夏普均值 {_fmt(wf['window_sharpe_mean'], nd=2)}。")
    L.append(f"- 参数稳健性区间（max−min）= {_fmt(ov['range'], nd=2)}，越小越稳健。")
    L.append("")
    L.append("> 本报告由 `kairos_strategies.validation` 自动生成，基于回测样本数据，"
             "仅用于方法演示与实现验证，**不构成任何投资建议**。")
    return "\n".join(L) + "\n"


def validate_strategy(data: MarketData, strategy: Strategy, out_dir: str,
                      n_windows: int = 4, n_boot: int = 500, ci: float = 0.95,
                      perturb: Sequence[float] = (0.5, 1.0, 2.0),
                      cost_rate: float = 0.001, seed: int = 0, block: int = 20,
                      max_evals: int = 60, mode: str = "full") -> Dict:
    """对单个策略做完整验证并落盘产物。

    汇总：基线回测 + walk-forward 样本外 + 参数敏感性(OAT) + PSR + bootstrap
    夏普置信区间。写出 `out_dir/report.md`（中文，含各表）与
    `out_dir/validation.json`（机器可读，NaN/inf→null）。返回汇总 dict
    （含扁平 `row` 字段，供批量汇总使用）。
    """
    os.makedirs(out_dir, exist_ok=True)
    bt = Backtester(cost_rate=cost_rate, risk_free=0.0,
                    periods_per_year=data.periods_per_year)
    res = bt.run(data, strategy.generate_weights(data))
    baseline = {k: float(v) for k, v in res.metrics.items()}
    wf = walk_forward_oos(data, strategy, n_windows=n_windows, cost_rate=cost_rate)
    sens = parameter_sensitivity(data, strategy, perturb=perturb,
                                 cost_rate=cost_rate, max_evals=max_evals)
    psr_d = _psr_components(res.returns, sr_benchmark=0.0)
    lo, hi = bootstrap_sharpe_ci(res.returns, n_boot=n_boot, ci=ci, block=block,
                                 seed=seed, periods_per_year=data.periods_per_year)
    span = ((str(data.dates[0].date()), str(data.dates[-1].date()))
            if len(data.dates) else ("NA", "NA"))
    summary: Dict[str, Any] = {
        "name": strategy.name,
        "channel": strategy.channel,
        "mode": str(mode),
        "params": {k: (v if isinstance(v, (int, float, str, bool)) or _is_numeric_param(v) else str(v))
                   for k, v in dict(strategy.params).items()},
        "data": {"name": data.name, "n_assets": int(data.n_assets),
                 "n_periods": int(len(data.dates)), "start": span[0], "end": span[1],
                 "periods_per_year": int(data.periods_per_year)},
        "cost_rate": float(cost_rate),
        "baseline": baseline,
        "walk_forward": wf,
        "sensitivity": sens,
        "psr": psr_d["psr"],
        "psr_detail": psr_d,
        "sharpe_ci": {"ci": float(ci), "lo": lo, "hi": hi,
                      "n_boot": int(n_boot), "block": int(block), "seed": int(seed),
                      "point_annualized_sharpe": baseline["sharpe"]},
    }
    summary["row"] = {
        "name": summary["name"], "channel": summary["channel"], "mode": str(mode),
        "error": None,
        "total_return": baseline["total_return"], "sharpe": baseline["sharpe"],
        "max_drawdown": baseline["max_drawdown"],
        "psr": psr_d["psr"], "sharpe_ci_lo": lo, "sharpe_ci_hi": hi,
        "positive_window_ratio": wf["positive_window_ratio"],
        "window_sharpe_mean": wf["window_sharpe_mean"],
        "window_sharpe_std": wf["window_sharpe_std"],
        "n_windows": wf["n_windows"],
        "sens_sharpe_min": sens["overall"]["min"],
        "sens_sharpe_median": sens["overall"]["median"],
        "sens_sharpe_max": sens["overall"]["max"],
        "sens_range": sens["overall"]["range"],
    }
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(_validation_md(summary))
    return summary


def _blank_row(strategy: Strategy, mode: str) -> Dict:
    """验证失败策略的占位汇总行。"""
    row = {"name": strategy.name, "channel": strategy.channel, "mode": mode, "error": None}
    for k in ("total_return", "sharpe", "max_drawdown", "psr", "sharpe_ci_lo", "sharpe_ci_hi",
              "positive_window_ratio", "window_sharpe_mean", "window_sharpe_std", "n_windows",
              "sens_sharpe_min", "sens_sharpe_median", "sens_sharpe_max", "sens_range"):
        row[k] = None
    return row


def _summary_md(rows: List[Dict], data: MarketData, mode: str, cfg: Dict) -> str:
    """VALIDATION_SUMMARY.md：全策略验证汇总表（按 PSR 降序，失败者置底）。"""
    ok = [r for r in rows if r.get("error") is None]
    bad = [r for r in rows if r.get("error") is not None]
    ok.sort(key=lambda r: r["psr"], reverse=True)
    ci_pct = cfg["ci"] * 100
    lines = [
        "# 样本外验证汇总 (VALIDATION_SUMMARY)",
        "",
        f"> 数据：{data.name}（{data.n_assets} 资产，{len(data.dates)} 期）｜ 模式：**{mode}**"
        f"（walk-forward {cfg['n_windows']} 窗、bootstrap {cfg['n_boot']} 次、OAT 扰动 ×0.5/×1.0/×2.0）。",
        "> 由 `kairos_strategies.validation` / `examples/validate.py` 自动生成，"
        f"**{'合成数据演示' if str(data.name).startswith('synthetic') else f'真实历史数据（{data.name}）回测'}，非投资建议**。按 PSR 降序排列。",
        "",
        f"| 策略 | 渠道 | 累计收益 | 夏普 | PSR | 夏普{ci_pct:.0f}%CI | 正收益窗口占比 "
        f"| 窗口夏普 均值±std | 敏感性区间(稳健性) | 敏感性中位夏普 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        lines.append(
            f"| [{r['name']}](validation/{r['name']}) | {r['channel']} "
            f"| {_fmt(r['total_return'], True)} | {_fmt(r['sharpe'], nd=2)} | {_fmt(r['psr'], nd=3)} "
            f"| [{_fmt(r['sharpe_ci_lo'], nd=2)}, {_fmt(r['sharpe_ci_hi'], nd=2)}] "
            f"| {_fmt(r['positive_window_ratio'], True)} "
            f"| {_fmt(r['window_sharpe_mean'], nd=2)} ± {_fmt(r['window_sharpe_std'], nd=2)} "
            f"| {_fmt(r['sens_sharpe_min'], nd=2)} ~ {_fmt(r['sens_sharpe_max'], nd=2)} "
            f"({_fmt(r['sens_range'], nd=2)}) | {_fmt(r['sens_sharpe_median'], nd=2)} |"
        )
    if bad:
        lines += ["", "## 验证失败的策略", ""]
        for r in bad:
            lines.append(f"- `{r['name']}`（{r['channel']}）：{r['error']}")
    lines += ["",
              f"合计策略：**{len(rows)}** 个（成功 {len(ok)} / 失败 {len(bad)}）。"
              "逐策略详情见 `validation/<name>/`（report.md + validation.json）。"]
    return "\n".join(lines) + "\n"


def run_validation(data: MarketData, strategies: List[Strategy], research_root: str,
                   mode: str = "quick", cost_rate: float = 0.001, seed: int = 0,
                   ci: float = 0.95,
                   perturb: Sequence[float] = (0.5, 1.0, 2.0)) -> List[Dict]:
    """批量验证多个策略：产物写 `research_root/validation/<name>/`，
    并汇总 `research_root/VALIDATION_SUMMARY.md`（按 PSR 降序）。

    mode="quick" 时减少 walk-forward 窗口数与 bootstrap 次数、并对敏感性评估
    次数设更低上限以加速；mode="full" 用默认强度。单策略验证失败不中断整体，
    失败行带 error 字段置底。返回汇总行列表（与输入顺序一致）。
    """
    is_full = str(mode).lower() == "full"
    mode_name = "full" if is_full else "quick"
    cfg = dict(FULL_CFG if is_full else QUICK_CFG)
    cfg["ci"] = float(ci)
    root = os.path.join(research_root, "validation")
    os.makedirs(root, exist_ok=True)
    rows: List[Dict] = []
    for s in strategies:
        out_dir = os.path.join(root, s.name)
        try:
            summary = validate_strategy(
                data, s, out_dir, n_windows=cfg["n_windows"], n_boot=cfg["n_boot"],
                ci=cfg["ci"], perturb=perturb, cost_rate=cost_rate, seed=seed,
                max_evals=cfg["max_evals"], mode=mode_name)
            rows.append(summary["row"])
        except Exception as exc:
            row = _blank_row(s, mode_name)
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
                json.dump(_jsonable(row), f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(research_root, "VALIDATION_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(_summary_md(rows, data, mode_name, cfg))
    return rows
