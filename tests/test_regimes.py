import numpy as np
import pandas as pd
import pytest

from forex.data import Panel
from forex.data.rates import with_retries
from forex.strategies import TrendGate, VolGate, XSEnsemble, XSMomentum
from forex.strategies.regimes import efficiency_ratio, stress_index


def _frame(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D", tz="UTC", name="time")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
            "bid": c - 0.0001, "ask": c + 0.0001, "volume": 100.0, "complete": True,
        },
        index=idx,
    )


def _calm_then_stressed(n_calm=500, n_stress=80, seed=0):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0, 0.001, n_calm)
    stress = rng.normal(0, 0.05, n_stress)
    return 1.0 * np.exp(np.cumsum(np.concatenate([calm, stress])))


def test_stress_index_rises_in_stressed_tail():
    panel = Panel({
        "AAA_USD": _frame(_calm_then_stressed(seed=1)),
        "BBB_USD": _frame(_calm_then_stressed(seed=2)),
    })
    stress = stress_index(panel.close, window=20)
    assert stress.iloc[-1] > 10 * stress.iloc[400]


def test_vol_gate_flattens_when_stressed():
    panel = Panel({
        "AAA_USD": _frame(_calm_then_stressed(seed=1)),
        "BBB_USD": _frame(_calm_then_stressed(seed=2)),
        "CCC_USD": _frame(_calm_then_stressed(seed=3)),
    })
    gated = VolGate(XSMomentum(40)).score(panel)
    raw = XSMomentum(40).score(panel)
    # Stressed tail: everything gated off.
    assert gated.iloc[-1].isna().all()
    # Calm mid-history (post warm-up): gate open, scores pass through.
    assert gated.iloc[450].notna().all()
    pd.testing.assert_series_equal(gated.iloc[450], raw.iloc[450])


def test_efficiency_ratio_separates_trend_from_chop():
    n = 200
    trend = np.linspace(1.0, 1.5, n)                      # straight line
    chop = 1.0 + 0.01 * np.array([(-1) ** i for i in range(n)])  # sawtooth
    close = pd.DataFrame({"TREND": trend, "CHOP": chop})
    er = efficiency_ratio(close, window=50)
    assert er["TREND"].iloc[-1] > 0.95
    assert er["CHOP"].iloc[-1] < 0.05


def test_trend_gate_keeps_trending_pair_only():
    n = 300
    trend = np.linspace(1.0, 1.6, n)
    chop = 1.0 + 0.01 * np.array([(-1) ** i for i in range(n)])
    panel = Panel({"TRD_USD": _frame(trend), "CHP_USD": _frame(chop)})
    gated = TrendGate(XSMomentum(40), window=50).score(panel)
    assert not np.isnan(gated["TRD_USD"].iloc[-1])
    assert np.isnan(gated["CHP_USD"].iloc[-1])


def test_gates_are_point_in_time():
    panel = Panel({
        "AAA_USD": _frame(_calm_then_stressed(seed=4)),
        "BBB_USD": _frame(_calm_then_stressed(seed=5)),
    })
    for gate in (VolGate(XSMomentum(40)), TrendGate(XSMomentum(40))):
        full = gate.score(panel).iloc[:400]
        trunc = gate.score(panel.upto(panel.index[399]))
        pd.testing.assert_frame_equal(full, trunc)


def test_ensemble_available_mean_lets_members_abstain():
    n = 300
    trend = np.linspace(1.0, 1.6, n)
    chop = 1.0 + 0.01 * np.array([(-1) ** i for i in range(n)])
    panel = Panel({"TRD_USD": _frame(trend), "CHP_USD": _frame(chop)})
    gated_member = TrendGate(XSMomentum(40), window=50)  # NaN on CHP
    always_member = XSMomentum(60)
    strict = XSEnsemble([gated_member, always_member]).score(panel)
    lenient = XSEnsemble([gated_member, always_member], require_all=False).score(panel)
    # Strict: CHP has no score (one member NaN). Lenient: CHP falls back to the
    # remaining member's z-score alone.
    assert np.isnan(strict["CHP_USD"].iloc[-1])
    assert not np.isnan(lenient["CHP_USD"].iloc[-1])
    # Where all members are present, the two modes agree exactly.
    both = strict["TRD_USD"].dropna().index
    pd.testing.assert_series_equal(
        strict.loc[both, "TRD_USD"], lenient.loc[both, "TRD_USD"]
    )


def test_registry_builds_gated_specs():
    from forex.strategies.registry import build_signal

    sig = build_signal({
        "type": "xs_ensemble",
        "require_all": False,
        "members": [
            {"type": "trend_gate", "member": {"type": "xsmom", "params": {"lookback": 120}}},
            {"type": "vol_gate", "member": {"type": "xscarry"}},
            {"type": "xsvalue", "params": {"lookback": 1000}},
        ],
    })
    assert isinstance(sig, XSEnsemble)
    assert not sig.require_all
    assert isinstance(sig.signals[0], TrendGate)
    assert isinstance(sig.signals[1], VolGate)


def test_gate_rejects_bad_mode():
    with pytest.raises(ValueError):
        VolGate(XSMomentum(20), active_when="sideways")
    with pytest.raises(ValueError):
        TrendGate(XSMomentum(20), active_when="sideways")


def test_with_retries_recovers_from_transient_failures():
    attempts = []

    def flaky(x):
        attempts.append(x)
        if len(attempts) < 3:
            raise TimeoutError("slow")
        return x * 2

    wrapped = with_retries(flaky, retries=3, base_delay=0.0)
    assert wrapped(21) == 42
    assert len(attempts) == 3

    def always_fails(_):
        raise TimeoutError("dead")

    with pytest.raises(TimeoutError):
        with_retries(always_fails, retries=2, base_delay=0.0)(1)
