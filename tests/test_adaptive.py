"""adaptive 渠道测试（在线学习 / 专家加权元策略）。

覆盖：
  - 契约与数值健全：形状/索引列对齐/有限值、long_only 权重 ≥ 0、每行权重和 ≤ 1+eps、确定性；
  - 元信息：channel='adaptive'、名字唯一且不与专家重名、无参可构造、只 import 其它渠道公开类；
  - 在线学习数值核对（测试内独立重算，不复用被测实现）：
      hedge_experts   系数 == softmax(η·shift1(累计对数奖励))；η=0 时退化为等权；
      bandit_select   预热期等权、其后每行 one-hot、选择只在段首变化、并列取池内首个；
      online_mom_rev_switch 系数 == 0.5 ± tilt_max·z/(1+|z|)（z 为 shift1 的相对优势）；
      adaptive_param  effective_windows / exposure_scale == 由 shift1 已实现波动独立重算；
  - 行为断言（regime 切换场景，专家A前段强、专家B后段强）：
      hedge/bandit/switch 的权重随时间从偏向 A 切换到偏向 B；专家权重非负且和≈1；
      adaptive_param 高波期窗口更短、敞口更低、权重更小；
  - 防未来函数：篡改 t0 之后价格，t0 及之前权重/系数逐位不变；截断样本前缀完全一致；
      只篡改第 t 期及之后奖励，第 t 期系数不变（显式验证 shift/lag）；
  - 性能预算与极小数据边界。
"""
import time

import numpy as np
import pandas as pd
import pytest

import kairos_strategies as ks
from kairos_strategies import MarketData
from kairos_strategies.channels import adaptive as ad
from kairos_strategies.channels.adaptive import (
    AdaptiveParamStrategy,
    BanditSelectStrategy,
    HedgeExpertsStrategy,
    OnlineMomRevSwitchStrategy,
)

STRATEGIES = [HedgeExpertsStrategy, BanditSelectStrategy,
              OnlineMomRevSwitchStrategy, AdaptiveParamStrategy]
POOLED = [HedgeExpertsStrategy, BanditSelectStrategy]     # 使用完整专家池的元策略
EPS = 1e-9
N_ASSETS, N_DAYS, SEED = 6, 400, 1
T0 = 300                      # 防未来篡改点（之后价格被剧烈改写）
PREFIX = 250                  # 前缀截断长度
SPLIT = 200                   # regime 切换点（前半趋势、后半震荡）


# ----------------------------------------------------------------------
# 数据夹具
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    return ks.make_synthetic_universe(n_assets=N_ASSETS, n_days=N_DAYS, seed=SEED)


@pytest.fixture(scope="module")
def regime():
    """regime 切换场景（纯确定性公式，无 RNG）：前半平滑上行利于趋势/配置专家，
    后半正弦震荡利于 RSI 均值回归专家——构造『专家A前段强、专家B后段强』。"""
    idx = pd.bdate_range("2018-01-02", periods=N_DAYS)
    t = np.arange(N_DAYS)
    prices = {}
    for i in range(4):
        phi = 2.0 * np.pi * i / 4.0
        g = 0.0006 * (1.0 + 0.15 * i)
        ret = np.zeros(N_DAYS)
        ret[:SPLIT] = g + 0.0015 * np.sin(2.0 * np.pi * t[:SPLIT] / 7.0 + phi)
        s = t[SPLIT:]
        ret[SPLIT:] = -0.0006 + 0.010 * (1.0 + 0.1 * i) * np.sin(2.0 * np.pi * s / 16.0 + phi)
        prices[f"A{i}"] = 100.0 * np.cumprod(1.0 + ret)
    return MarketData(prices=pd.DataFrame(prices, index=idx), volumes=None,
                      periods_per_year=252, name="regime_switch")


def _tamper_future(data: MarketData, t0: int = T0, seed: int = 99) -> MarketData:
    """把 t0 之后的价格乘上 [2.5, 3.0) 的随机因子（剧烈改写未来，价格仍恒正）。"""
    rng = np.random.default_rng(seed)
    p = data.prices.copy()
    fut = p.iloc[t0 + 1:]
    p.iloc[t0 + 1:] = fut * (2.5 + 0.5 * rng.random(fut.shape))
    return MarketData(prices=p,
                      volumes=None if data.volumes is None else data.volumes.copy(),
                      periods_per_year=data.periods_per_year, name=data.name)


def _truncate(data: MarketData, m: int) -> MarketData:
    return MarketData(prices=data.prices.iloc[:m],
                      volumes=None if data.volumes is None else data.volumes.iloc[:m],
                      periods_per_year=data.periods_per_year, name=data.name)


def _regime_winners(data: MarketData):
    """返回 (A=前半段累计收益最高的专家, B=后半段最强的均值回归专家)。"""
    rets = ad.expert_backtest_returns(data)
    p1 = np.expm1(np.log1p(rets.iloc[:SPLIT]).sum())
    p2 = np.expm1(np.log1p(rets.iloc[SPLIT:]).sum())
    A = str(p1.idxmax())
    B = "rsi_reversion"
    assert A != B and str(p2.idxmax()) == B       # 场景有效性：B 确实是后半段赢家
    return A, B


# ----------------------------------------------------------------------
# 契约与数值健全
# ----------------------------------------------------------------------

def test_shape_columns_index_alignment(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert isinstance(w, pd.DataFrame), cls.name
        assert w.shape == (len(synth.dates), len(synth.symbols)), cls.name
        assert list(w.columns) == synth.symbols, cls.name
        assert w.index.equals(synth.dates), cls.name


def test_all_values_finite(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert np.isfinite(w.to_numpy()).all(), cls.name


def test_long_only_weights_non_negative(synth):
    for cls in STRATEGIES:
        s = cls()
        assert s.long_only is True
        w = s.generate_weights(synth)
        assert w.to_numpy().min() >= 0.0, cls.name


def test_row_sum_le_one(synth):
    for cls in STRATEGIES:
        w = cls().generate_weights(synth)
        assert w.abs().sum(axis=1).max() <= 1.0 + EPS, cls.name
        assert w.to_numpy().max() <= 1.0 + EPS, cls.name


def test_deterministic_same_input_same_output(synth):
    for cls in STRATEGIES:
        w1 = cls().generate_weights(synth)
        w2 = cls().generate_weights(synth)          # 新实例、二次调用
        pd.testing.assert_frame_equal(w1, w2)
        assert np.array_equal(w1.to_numpy(), w2.to_numpy()), cls.name


def test_no_arg_construction_and_meta_complete():
    names = set()
    expert_names = set(ad.expert_base_names())
    for cls in STRATEGIES:
        s = cls()                                   # 必须无参可构造
        meta = s.meta()
        assert s.channel == "adaptive"
        assert s.name and s.name == s.name.strip().lower()
        assert s.name not in names and s.name not in expert_names, s.name
        names.add(s.name)
        assert s.long_only is True
        assert s.universe in ("timing", "cross_section")
        for key in ("description", "hypothesis", "source"):
            assert isinstance(meta[key], str) and len(meta[key]) >= 20, (s.name, key)
        assert isinstance(meta["params"], dict) and meta["params"], s.name


def test_only_public_base_classes_imported():
    """只允许 import 其它渠道的公开 Strategy 类：不得引入其私有下划线函数。"""
    prefix = "kairos_strategies.channels."
    foreign = {k: v for k, v in vars(ad).items()
               if getattr(v, "__module__", None) and
               str(getattr(v, "__module__")).startswith(prefix) and
               getattr(v, "__module__") != ad.__name__}
    assert foreign, "本渠道应复用其它渠道的基础策略"
    for nm, obj in foreign.items():
        assert not nm.startswith("_"), f"不应以私有名暴露外部符号: {nm}"
        assert isinstance(obj, type) and issubclass(obj, ks.Strategy), nm
        assert getattr(obj, "channel", "") != "adaptive", nm


# ----------------------------------------------------------------------
# 专家池 / 奖励回测
# ----------------------------------------------------------------------

def test_expert_pool_is_expected_strategies():
    assert ad.expert_base_names() == ["sma_cross", "donchian_turtle", "rsi_reversion",
                                      "ts_momentum", "tsmom_volscaled", "low_volatility",
                                      "inverse_vol"]
    strats = ad.expert_strategies()
    assert list(strats) == ad.expert_base_names()
    for nm, s in strats.items():
        assert s.name == nm
        assert s.channel in ("technical", "momentum", "trend", "factor", "allocation")
    assert ad.switch_leg_names() == ("ts_momentum", "rsi_reversion")
    for leg in ad.switch_leg_names():
        assert leg in set(ad.expert_base_names())


def test_expert_panels_long_only_and_aligned(synth):
    panels = ad.expert_weight_panels(synth)
    assert set(panels) == set(ad.expert_base_names())
    for nm, p in panels.items():
        assert p.shape == (N_DAYS, N_ASSETS), nm
        assert list(p.columns) == synth.symbols, nm
        assert np.isfinite(p.to_numpy()).all(), nm
        assert p.to_numpy().min() >= 0.0, nm                    # 统一成 long_only
        assert p.sum(axis=1).max() <= 1.0 + EPS, nm


def test_expert_returns_match_manual_lagged_computation(synth):
    """专家奖励 == 手工「上一期权重 × 本期资产收益」（t 期期末才可知）。"""
    rets = ad.expert_backtest_returns(synth)
    names = ad.expert_base_names()
    assert list(rets.columns) == names
    assert rets.index.equals(synth.dates)
    assert np.isfinite(rets.to_numpy()).all()
    asset_ret = synth.prices.pct_change().fillna(0.0)
    panels = ad.expert_weight_panels(synth, names=names)
    for nm in names:
        manual = (panels[nm].shift(1).fillna(0.0) * asset_ret).sum(axis=1)
        assert np.allclose(rets[nm].to_numpy(), manual.to_numpy(), atol=1e-15), nm


# ----------------------------------------------------------------------
# hedge_experts —— Hedge/指数权重（数值核对）
# ----------------------------------------------------------------------

def _expected_hedge(rets: pd.DataFrame, eta: float) -> np.ndarray:
    """测试内独立重算 Hedge 系数：softmax(η·下移一行的累计对数奖励)。"""
    vals = np.clip(np.nan_to_num(rets.to_numpy(dtype="float64")), -0.999, None)
    g = np.log1p(vals)
    G = np.cumsum(g, axis=0)                     # decay=1 即普通累计
    prev = np.zeros_like(G)
    prev[1:] = G[:-1]                            # 防未来：第 t 期用 G[≤t-1]
    z = eta * prev
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_hedge_coefficients_match_manual_recomputation(synth):
    s = HedgeExpertsStrategy()
    rets = ad.expert_backtest_returns(synth)
    coef = s.mix_coefficients(synth)
    assert list(coef.columns) == ad.expert_base_names()
    expected = _expected_hedge(rets, float(s.params["eta"]))
    assert np.allclose(coef.to_numpy(), expected, atol=1e-12, rtol=0.0)


def test_hedge_coefficients_nonneg_sum_one_and_first_equal(synth):
    s = HedgeExpertsStrategy()
    coef = s.mix_coefficients(synth).to_numpy()
    assert coef.min() >= 0.0
    assert np.allclose(coef.sum(axis=1), 1.0, atol=1e-12)
    k = len(ad.expert_base_names())
    assert np.allclose(coef[0], 1.0 / k, atol=1e-12)        # t=0 无历史 → 等权


def test_hedge_eta_zero_reduces_to_equal_weight(synth):
    s = HedgeExpertsStrategy()
    s.params = {**s.params, "eta": 0.0}                      # 不修改类属性
    coef = s.mix_coefficients(synth).to_numpy()
    k = len(ad.expert_base_names())
    assert np.allclose(coef, 1.0 / k, atol=1e-12)


# ----------------------------------------------------------------------
# bandit_select —— 多臂老虎机（结构核对）
# ----------------------------------------------------------------------

def test_bandit_warmup_equal_then_onehot(synth):
    s = BanditSelectStrategy()
    win = int(s.params["window"])
    coef = s.mix_coefficients(synth).to_numpy()
    k = len(ad.expert_base_names())
    assert np.allclose(coef[:win], 1.0 / k, atol=1e-12)      # 预热期等权
    live = coef[win:]
    assert np.allclose(live.sum(axis=1), 1.0, atol=1e-12)    # 每行和为 1
    assert np.allclose(live.max(axis=1), 1.0, atol=1e-12)    # one-hot
    assert (np.isclose(live, 1.0).sum(axis=1) == 1).all()    # 恰一个 1


def test_bandit_selection_changes_only_at_block_starts(synth):
    s = BanditSelectStrategy()
    win, hold = int(s.params["window"]), int(s.params["hold"])
    sel = s.selections(synth).to_numpy()
    assert (sel[:win] == "").all()                           # 预热期空串
    for t in range(win + 1, len(sel)):
        if sel[t] != sel[t - 1]:
            assert (t - win) % hold == 0, t                  # 只在段首切换


def test_bandit_tie_break_is_deterministic_first_expert():
    """并列（所有专家奖励相同）时确定性选中池内首个专家（argmax 取首下标）。"""
    n, k = 120, len(ad.expert_base_names())
    rets = pd.DataFrame(np.full((n, k), 0.001),
                        index=pd.bdate_range("2020-01-01", periods=n),
                        columns=ad.expert_base_names())
    coef, sel = ad._bandit_coefficients(rets, window=10, hold=5, explore_c=0.0)
    coef = coef.to_numpy()
    first = ad.expert_base_names()[0]
    assert (sel[10:] == 0).all()                             # 下标恒为 0
    labels = [first if i >= 0 else "" for i in sel]
    assert all(x == first for x in labels[10:])
    assert np.allclose(coef[10:, 0], 1.0)


# ----------------------------------------------------------------------
# online_mom_rev_switch —— 在线切换（数值核对）
# ----------------------------------------------------------------------

def test_switch_coefficients_match_manual_recomputation(synth):
    s = OnlineMomRevSwitchStrategy()
    win, tmax, floor = int(s.params["window"]), float(s.params["tilt_max"]), float(s.params["std_floor"])
    rets = ad.expert_backtest_returns(synth)
    mom, rev = ad.switch_leg_names()
    diff = rets[mom] - rets[rev]
    roll = diff.rolling(win, min_periods=win)
    z = (roll.mean() / roll.std(ddof=0).clip(lower=floor)).shift(1).fillna(0.0).to_numpy()
    tilt = tmax * z / (1.0 + np.abs(z))
    expected = np.stack([0.5 + tilt, 0.5 - tilt], axis=1)
    coef = s.leg_coefficients(synth)
    assert list(coef.columns) == ["momentum", "reversion"]
    assert np.allclose(coef.to_numpy(), expected, atol=1e-12, rtol=0.0)


def test_switch_coefficients_bounded_and_sum_one(synth):
    s = OnlineMomRevSwitchStrategy()
    tmax = float(s.params["tilt_max"])
    coef = s.leg_coefficients(synth).to_numpy()
    assert coef.min() >= 0.0
    assert np.allclose(coef.sum(axis=1), 1.0, atol=1e-12)
    assert coef.max() <= 0.5 + tmax + EPS
    assert coef.min() >= 0.5 - tmax - EPS
    win = int(s.params["window"])
    assert np.allclose(coef[:win], 0.5, atol=1e-12)          # 预热期中性 50/50


# ----------------------------------------------------------------------
# adaptive_param —— 在线自适应参数（数值核对 + 行为）
# ----------------------------------------------------------------------

def _expected_sigma(data: MarketData, s: AdaptiveParamStrategy) -> pd.DataFrame:
    p = s.params
    rets = data.prices.astype("float64").pct_change().fillna(0.0)
    vw = max(int(p["vol_window"]), 2)
    rv = (rets.rolling(vw, min_periods=vw).std()
          * np.sqrt(float(data.periods_per_year))).shift(1)   # 防未来：≤t-1
    return rv.replace([np.inf, -np.inf], np.nan).fillna(float(p["ref_vol"])).clip(lower=float(p["vol_floor"]))


def test_adaptive_param_windows_and_exposure_match_manual(synth):
    s = AdaptiveParamStrategy()
    p = s.params
    sigma = _expected_sigma(synth, s)
    W = (float(p["base_window"]) * float(p["ref_vol"]) / sigma).round()
    W = W.clip(lower=int(p["min_window"]), upper=int(p["max_window"])).astype(int)
    E = (float(p["target_vol"]) / sigma).clip(lower=0.0, upper=1.0)
    assert (s.effective_windows(synth).to_numpy() == W.to_numpy()).all()
    assert np.allclose(s.exposure_scale(synth).to_numpy(), E.to_numpy(), atol=1e-12)


def test_adaptive_param_window_bounds_and_exposure_range(synth):
    s = AdaptiveParamStrategy()
    p = s.params
    W = s.effective_windows(synth).to_numpy()
    E = s.exposure_scale(synth).to_numpy()
    assert W.min() >= int(p["min_window"]) and W.max() <= int(p["max_window"])
    assert E.min() >= 0.0 and E.max() <= 1.0 + EPS


def test_adaptive_param_high_vol_shorter_window_lower_exposure():
    """行为：低波前半 → 长窗口/高敞口；高波后半 → 短窗口/低敞口/更低权重。"""
    idx = pd.bdate_range("2019-01-01", periods=400)
    t = np.arange(400)
    r = np.concatenate([0.0004 + 0.002 * np.sin(2 * np.pi * t[:200] / 30.0),
                        0.0004 + 0.05 * np.sin(2 * np.pi * t[200:] / 5.0)])
    data = MarketData(prices=pd.DataFrame({"A0": 100.0 * np.cumprod(1 + r)}, index=idx),
                      volumes=None, name="hv")
    s = AdaptiveParamStrategy()
    W = s.effective_windows(data)["A0"].to_numpy()
    E = s.exposure_scale(data)["A0"].to_numpy()
    w = s.generate_weights(data)["A0"].to_numpy()
    early = slice(60, 190)                                    # 低波段（预热后）
    late = slice(260, 400)                                    # 高波段
    assert W[late].mean() < W[early].mean()                   # 高波 → 更短窗口
    assert E[late].mean() < E[early].mean()                   # 高波 → 更低敞口
    assert w[late].mean() < w[early].mean()                   # 敞口更低 → 权重更小


# ----------------------------------------------------------------------
# 行为断言：regime 切换场景（专家A前段强、专家B后段强）
# ----------------------------------------------------------------------

def test_regime_weights_shift_from_A_to_B(regime):
    A, B = _regime_winners(regime)

    # hedge_experts：早段偏向 A，晚段偏向 B（且 B 成为晚段 argmax）
    h = HedgeExpertsStrategy()
    coef = h.mix_coefficients(regime)
    mid, late = coef.iloc[SPLIT - 60:SPLIT], coef.iloc[-40:]
    assert mid[A].mean() > mid[B].mean()                      # 前半倾斜向 A
    assert late[B].mean() > late[A].mean()                    # 后半倾斜向 B
    assert str(late.mean().idxmax()) == B                     # 晚段头号专家 = B
    assert coef[B].iloc[-40:].mean() > coef[B].iloc[:SPLIT].mean()   # B 权重显著上升

    # bandit_select：前半从不选 B，后半稳定切到 B
    b = BanditSelectStrategy()
    sel = b.selections(regime)
    pre = sel.iloc[int(b.params["window"]):SPLIT]
    assert (pre != B).all() and (pre != "").all()             # 预热后至切换前从不选 B
    assert (sel.iloc[-40:] == B).all()                        # 后半一路持有 B

    # online_mom_rev_switch：前半偏向动量腿，后半偏向回归腿
    s = OnlineMomRevSwitchStrategy()
    lc = s.leg_coefficients(regime)
    assert lc["momentum"].iloc[SPLIT - 60:SPLIT].mean() > lc["reversion"].iloc[SPLIT - 60:SPLIT].mean()
    assert lc["reversion"].iloc[SPLIT + 100:].mean() > lc["momentum"].iloc[SPLIT + 100:].mean()


def test_regime_expert_weights_nonneg_and_sum_one(regime):
    h = HedgeExpertsStrategy().mix_coefficients(regime).to_numpy()
    b = BanditSelectStrategy().mix_coefficients(regime).to_numpy()
    s = OnlineMomRevSwitchStrategy().leg_coefficients(regime).to_numpy()
    for coef in (h, b, s):
        assert coef.min() >= 0.0
        assert np.allclose(coef.sum(axis=1), 1.0, atol=1e-12)


def test_pooled_strategies_really_differ(synth):
    """hedge 与 bandit 的输出不同（软加权 vs 硬切换），且都区别于等权混合。"""
    wh = HedgeExpertsStrategy().generate_weights(synth).to_numpy()
    wb = BanditSelectStrategy().generate_weights(synth).to_numpy()
    assert not np.allclose(wh, wb, atol=1e-8)


# ----------------------------------------------------------------------
# 防未来函数
# ----------------------------------------------------------------------

def test_no_lookahead_tamper_future_prices(synth):
    """篡改 t0 之后的价格，t0 及之前的权重必须逐位完全不变（四个策略）。"""
    tampered = _tamper_future(synth, T0)
    assert not np.allclose(tampered.prices.to_numpy()[T0 + 1:],
                           synth.prices.to_numpy()[T0 + 1:])
    for cls in STRATEGIES:
        s = cls()
        w_full = s.generate_weights(synth).to_numpy()
        w_tam = s.generate_weights(tampered).to_numpy()
        assert np.array_equal(w_full[:T0 + 1], w_tam[:T0 + 1]), s.name
        assert not np.array_equal(w_full, w_tam), s.name      # 未来确实被改写了


def test_no_lookahead_prefix_truncation(synth):
    """截断样本重算，前缀权重必须与全样本逐位完全一致。"""
    sub = _truncate(synth, PREFIX)
    for cls in STRATEGIES:
        s = cls()
        full = s.generate_weights(synth).to_numpy()[:PREFIX]
        part = s.generate_weights(sub).to_numpy()
        assert np.array_equal(full, part), s.name


def test_no_lookahead_coefficient_prefix(synth):
    """元策略系数（hedge/bandit/switch）的前缀不变性。"""
    sub = _truncate(synth, PREFIX)
    h_full = HedgeExpertsStrategy().mix_coefficients(synth).to_numpy()[:PREFIX]
    h_part = HedgeExpertsStrategy().mix_coefficients(sub).to_numpy()
    assert np.array_equal(h_full, h_part)
    b_full = BanditSelectStrategy().selections(synth).to_numpy()[:PREFIX]
    b_part = BanditSelectStrategy().selections(sub).to_numpy()
    assert np.array_equal(b_full, b_part)
    s_full = OnlineMomRevSwitchStrategy().leg_coefficients(synth).to_numpy()[:PREFIX]
    s_part = OnlineMomRevSwitchStrategy().leg_coefficients(sub).to_numpy()
    assert np.array_equal(s_full, s_part)


def test_no_lookahead_hedge_ignores_current_period_reward(synth):
    """显式验证 shift(1)：只篡改第 t 期（含）之后的奖励，第 t 期系数不变。"""
    s = HedgeExpertsStrategy()
    rets = ad.expert_backtest_returns(synth)
    t = 200
    bumped = rets.copy()
    bumped.iloc[t:] = bumped.iloc[t:] + 0.05                  # 大幅改写 t 期及之后奖励
    base = ad._hedge_coefficients(rets, float(s.params["eta"]), float(s.params["decay"]))
    part = ad._hedge_coefficients(bumped, float(s.params["eta"]), float(s.params["decay"]))
    assert np.array_equal(base.iloc[:t].to_numpy(), part.iloc[:t].to_numpy())   # ≤t-1 不变
    assert not np.array_equal(base.to_numpy(), part.to_numpy())                 # ≥t 改变


def test_no_lookahead_switch_ignores_current_period_reward(synth):
    """切换器 z 值 shift(1)：篡改第 t 期及之后奖励差，第 t 期系数不变。"""
    s = OnlineMomRevSwitchStrategy()
    rets = ad.expert_backtest_returns(synth)
    mom, rev = ad.switch_leg_names()
    diff = (rets[mom] - rets[rev]).rename("diff")
    t = 200
    bumped = diff.copy()
    bumped.iloc[t:] = bumped.iloc[t:] + 0.05
    base = ad._switch_coefficients(diff, int(s.params["window"]),
                                   float(s.params["tilt_max"]), float(s.params["std_floor"]))
    part = ad._switch_coefficients(bumped, int(s.params["window"]),
                                   float(s.params["tilt_max"]), float(s.params["std_floor"]))
    assert np.array_equal(base.iloc[:t + 1].to_numpy(), part.iloc[:t + 1].to_numpy())
    assert not np.array_equal(base.to_numpy(), part.to_numpy())


# ----------------------------------------------------------------------
# 性能 / 边界
# ----------------------------------------------------------------------

def test_performance_budget_full_size_universe():
    """1000×8 数据上四个策略（含 7 次专家回测，缓存共享）应在数秒内完成。"""
    big = ks.make_synthetic_universe(n_assets=8, n_days=1000, seed=2026)
    t0 = time.perf_counter()
    for cls in STRATEGIES:
        w = cls().generate_weights(big)
        assert w.shape == (1000, 8)
        assert w.sum(axis=1).max() <= 1.0 + EPS
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"耗时 {elapsed:.2f}s 超出预算"


def test_tiny_dataset_edge_case():
    """极小数据（5 期 × 1 资产）不崩溃：全程预热回退，约束仍成立。"""
    tiny = MarketData(prices=pd.DataFrame(
        {"A0": np.array([100.0, 101.0, 99.0, 102.0, 103.0])},
        index=pd.bdate_range("2020-01-01", periods=5)))
    for cls in STRATEGIES:
        w = cls().generate_weights(tiny)
        assert w.shape == (5, 1), cls.name
        assert np.isfinite(w.to_numpy()).all(), cls.name
        assert w.to_numpy().min() >= 0.0, cls.name
        assert w.sum(axis=1).max() <= 1.0 + EPS, cls.name
