"""validation 模块（样本外验证 / 稳健性工具）的测试。"""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from kairos_strategies import make_synthetic_universe, metrics
from kairos_strategies import validation as val
from kairos_strategies.channels.benchmark import EqualWeightBuyHold
from kairos_strategies.channels.technical import SmaCrossStrategy


@pytest.fixture(scope="module")
def data():
    return make_synthetic_universe(n_assets=4, n_days=300, seed=7)


def test_walk_forward_window_count_and_metrics(data):
    out = val.walk_forward_oos(data, SmaCrossStrategy(), n_windows=4)
    assert out["n_windows"] == 4
    assert len(out["windows"]) == 4
    for w in out["windows"]:
        assert w["n_periods"] > 0
        assert math.isfinite(w["total_return"])
        assert math.isfinite(w["sharpe"])
        assert math.isfinite(w["volatility"])
        assert math.isfinite(w["max_drawdown"])
    assert 0.0 <= out["positive_window_ratio"] <= 1.0
    assert math.isfinite(out["window_sharpe_mean"])
    assert out["window_sharpe_std"] >= 0.0
    assert out["worst_window_return"] <= out["best_window_return"]
    assert out["windows"][0]["start"] == str(data.dates[0].date())
    assert out["windows"][-1]["end"] == str(data.dates[-1].date())


def test_parameter_sensitivity_maps_order_and_no_pollution(data):
    s = SmaCrossStrategy()
    before = copy.deepcopy(s.params)
    out = val.parameter_sensitivity(data, s, perturb=(0.5, 1.0, 2.0))
    assert s.params == before                       # 原实例 params 未被污染
    assert SmaCrossStrategy.params == before        # 类属性也未被污染
    assert set(out["params"]) == {"fast", "slow", "band"}
    fast_vals = [p["value"] for p in out["params"]["fast"]["points"]]
    assert 5 in fast_vals and 10 in fast_vals and 20 in fast_vals   # int 扰动 ×0.5/×1/×2
    for key, info in out["params"].items():
        assert info["points"]
        assert info["value_sharpe"]
        for p in info["points"]:
            assert math.isfinite(p["sharpe"])
        assert info["min"] <= info["median"] <= info["max"]
    ov = out["overall"]
    assert ov["min"] <= ov["median"] <= ov["max"]
    assert ov["range"] == pytest.approx(ov["max"] - ov["min"])
    assert ov["min"] <= out["baseline_sharpe"] <= ov["max"]


def test_parameter_sensitivity_no_numeric_params(data):
    out = val.parameter_sensitivity(data, EqualWeightBuyHold(), max_evals=4)
    assert out["params"] == {}
    assert out["overall"]["range"] == pytest.approx(0.0)


def test_probabilistic_sharpe_high_sharpe_near_one():
    rng = np.random.default_rng(42)
    r = pd.Series(0.004 + 0.01 * rng.standard_normal(1000))
    psr = val.probabilistic_sharpe(r)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.95


def test_probabilistic_sharpe_zero_mean_near_half():
    r = pd.Series(np.tile([0.01, -0.01], 200))
    assert val.probabilistic_sharpe(r) == pytest.approx(0.5, abs=1e-9)
    rng = np.random.default_rng(0)
    r2 = pd.Series(rng.standard_normal(2000))
    r2 = r2 - r2.mean()
    assert val.probabilistic_sharpe(r2) == pytest.approx(0.5, abs=1e-6)
    assert val.probabilistic_sharpe(r2, sr_benchmark=0.05) < 0.5


def test_probabilistic_sharpe_always_in_unit_interval():
    for r in ([0.01] * 5, [0.0, 0.0, 0.0, 0.0], np.linspace(-0.05, 0.05, 9)):
        psr = val.probabilistic_sharpe(np.asarray(r, dtype=float))
        assert 0.0 <= psr <= 1.0


def test_bootstrap_sharpe_ci_brackets_point_and_deterministic():
    rng = np.random.default_rng(3)
    r = pd.Series(0.0008 + 0.01 * rng.standard_normal(800))
    lo1, hi1 = val.bootstrap_sharpe_ci(r, n_boot=300, ci=0.95, block=20, seed=0)
    lo2, hi2 = val.bootstrap_sharpe_ci(r, n_boot=300, ci=0.95, block=20, seed=0)
    assert (lo1, hi1) == (lo2, hi2)                 # 固定 seed 确定性
    assert lo1 <= hi1
    point = metrics.sharpe(r)                       # 年化夏普点估计
    assert lo1 - 0.25 <= point <= hi1 + 0.25


def test_bootstrap_sharpe_ci_wider_with_higher_ci_and_short_input():
    rng = np.random.default_rng(5)
    r = pd.Series(0.0005 + 0.01 * rng.standard_normal(500))
    lo90, hi90 = val.bootstrap_sharpe_ci(r, n_boot=200, ci=0.90, seed=1)
    lo99, hi99 = val.bootstrap_sharpe_ci(r, n_boot=200, ci=0.99, seed=1)
    assert lo99 <= lo90 and hi90 <= hi99
    assert val.bootstrap_sharpe_ci(np.array([0.01]), seed=0) == (0.0, 0.0)
    lo, hi = val.bootstrap_sharpe_ci(r.iloc[:10], n_boot=50, block=20, seed=2)  # block 自动钳制
    assert lo <= hi


def test_validate_strategy_writes_report_and_json(tmp_path, data):
    out_dir = str(tmp_path / "v")
    summary = val.validate_strategy(data, SmaCrossStrategy(), out_dir,
                                    n_windows=3, n_boot=50, max_evals=8)
    md = tmp_path / "v" / "report.md"
    js = tmp_path / "v" / "validation.json"
    assert md.exists() and js.exists()
    text = md.read_text(encoding="utf-8")
    assert "Walk-Forward" in text and "PSR" in text and "敏感性" in text
    j = json.loads(js.read_text(encoding="utf-8"))
    assert j["name"] == "sma_cross"
    assert 0.0 <= j["psr"] <= 1.0
    assert j["sharpe_ci"]["lo"] <= j["sharpe_ci"]["hi"]
    assert summary["walk_forward"]["n_windows"] == 3
    assert summary["row"]["psr"] == pytest.approx(summary["psr"])
    assert summary["row"]["error"] is None


def test_run_validation_writes_summary(tmp_path, data):
    strategies = [EqualWeightBuyHold(), SmaCrossStrategy()]
    rows = val.run_validation(data, strategies, str(tmp_path), mode="quick")
    assert len(rows) == 2
    summary_path = tmp_path / "VALIDATION_SUMMARY.md"
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    for s in strategies:
        assert s.name in text
        d = tmp_path / "validation" / s.name
        assert (d / "report.md").exists()
        assert (d / "validation.json").exists()
    for r in rows:
        assert r["error"] is None
        assert 0.0 <= r["psr"] <= 1.0
        assert r["sharpe_ci_lo"] <= r["sharpe_ci_hi"]
        assert 0.0 <= r["positive_window_ratio"] <= 1.0
        assert r["sens_range"] >= 0.0
        assert math.isfinite(r["sharpe"])
