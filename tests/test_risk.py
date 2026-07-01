import math

import numpy as np
import pandas as pd

from forex.backtest import expand_grid, walk_forward
from forex.backtest.engine import BacktestResult
from forex.backtest.evaluation import Backtestable
from forex.backtest.metrics import compute_metrics
from forex.risk import RiskOverlay, fixed_fractional_units, risk_overlay_factory


class StubBT(Backtestable):
    """Backtestable wrapping a fixed return series, for controlled tests."""

    granularity = "D"
    name = "stub"

    def __init__(self, returns: pd.Series):
        self.rets = returns

    def run(self, start=None, end=None):
        r = self.rets
        if end is not None:
            r = r[r.index <= end]
        if start is not None:
            r = r[r.index >= start]
        pos = pd.Series(1.0, index=r.index)
        return BacktestResult(
            returns=r,
            equity=(1 + r).cumprod(),
            position=pos,
            metrics=compute_metrics(r, pos, "D"),
            granularity="D",
        )


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="time")


def test_fixed_fractional_units():
    # $10k equity, risk 1%, 50-pip stop on EURUSD at 1.1000 -> 100/0.0050 = 20k units
    units = fixed_fractional_units(10_000, 0.01, 1.1000, 1.0950)
    assert abs(units - 20_000) < 1e-6


def test_vol_targeting_brings_vol_near_target():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.02, 1500), index=_idx(1500))  # ~32% ann vol
    target = 0.10
    overlay = RiskOverlay(StubBT(r), target_ann_vol=target, vol_window=60)
    res = overlay.run()
    realized = res.returns.std(ddof=1) * math.sqrt(252)
    assert abs(realized - target) / target < 0.3  # within 30% of target


def test_vol_scale_is_point_in_time():
    """Scale for bar t must not use bar t's own return."""
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0, 0.01, 400), index=_idx(400))
    overlay = RiskOverlay(StubBT(r), target_ann_vol=0.10, vol_window=30)
    scale = overlay._vol_scale(r)
    # Recompute by hand for a late bar: vol through t-1 only.
    t = 300
    manual = 0.10 / (r.iloc[t - 30 : t].std(ddof=1) * math.sqrt(252))
    manual = min(manual, overlay.max_leverage)
    assert abs(scale.iloc[t] - manual) < 1e-12


def test_kill_switch_caps_drawdown():
    # A relentless -1%/bar crash: without the switch dd -> -60%+; with a 10%
    # switch the overlay flattens once its own dd breaches -10%.
    crash = pd.Series([-0.01] * 100, index=_idx(100))
    overlay = RiskOverlay(StubBT(crash), max_drawdown=0.10)
    res = overlay.run()
    dd = (res.equity / res.equity.cummax() - 1.0).min()
    assert dd > -0.13  # breach detected within a couple of bars of -10%
    assert res.returns.iloc[-1] == 0.0  # flat by the end


def test_kill_switch_resumes_after_recovery():
    # Crash 15 bars at -1%, then steady +1% recovery: gate must re-open once the
    # underlying recovers resume_drawdown (default max_dd/2 = 5%) off its trough.
    r = pd.Series([-0.01] * 15 + [0.01] * 60, index=_idx(75))
    overlay = RiskOverlay(StubBT(r), max_drawdown=0.10)
    res = overlay.run()
    assert (res.returns.iloc[-20:] > 0).all()  # participating again
    assert overlay.last_gate == 1.0


def test_overlay_plugs_into_walk_forward():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0003, 0.01, 800), index=_idx(800))

    def base_factory(cfg):
        return StubBT(r)

    factory = risk_overlay_factory(base_factory, target_ann_vol=0.10)
    wf = walk_forward(factory, expand_grid({}), train=250, test=60)
    assert len(wf.folds) == (800 - 250) // 60
