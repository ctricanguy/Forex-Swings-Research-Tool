import numpy as np
import pandas as pd
import pytest

from forex.data import SyntheticDataProvider
from forex.signals import Direction
from forex.strategies import (
    Carry,
    Ensemble,
    MovingAverageCross,
    TimeSeriesMomentum,
)

ALL_PRICE_STRATEGIES = [
    TimeSeriesMomentum(lookback=20),
    MovingAverageCross(fast=10, slow=30),
]


def _data(n=300, **kw):
    return SyntheticDataProvider(**kw).get_candles("EUR_USD", "D", count=n)


def test_momentum_positions_match_trailing_return():
    data = _data()
    strat = TimeSeriesMomentum(lookback=20)
    pos = strat.generate_signals(data)["position"]
    trailing = data["close"] / data["close"].shift(20) - 1.0
    # Where trailing return is positive and defined, expect long; negative short.
    defined = trailing.notna()
    assert (pos[defined & (trailing > 0)] == 1.0).all()
    assert (pos[defined & (trailing < 0)] == -1.0).all()
    # Flat during the warm-up window.
    assert (pos[trailing.isna()] == 0.0).all()


def test_positions_in_range():
    data = _data()
    for strat in ALL_PRICE_STRATEGIES:
        pos = strat.generate_signals(data)["position"]
        assert pos.between(-1.0, 1.0).all()


@pytest.mark.parametrize("strat", ALL_PRICE_STRATEGIES, ids=lambda s: s.name)
def test_point_in_time_no_future_leak(strat):
    """Appending future bars must not change earlier positions."""
    full = _data(n=300)
    truncated = full.iloc[:200]
    pos_full = strat.generate_signals(full)["position"].iloc[:200]
    pos_trunc = strat.generate_signals(truncated)["position"]
    pd.testing.assert_series_equal(pos_full, pos_trunc)


def test_carry_requires_rate_diff():
    data = _data()  # no rate_diff column
    with pytest.raises(ValueError):
        Carry().generate_signals(data)


def test_carry_follows_rate_differential():
    data = _data(with_rate_diff=True)
    pos = Carry(threshold=0.25).generate_signals(data)["position"]
    diff = data["rate_diff"]
    assert (pos[diff > 0.25] == 1.0).all()
    assert (pos[diff < -0.25] == -1.0).all()


def test_ensemble_averages_members():
    data = _data(with_rate_diff=True)
    members = [TimeSeriesMomentum(lookback=20), Carry()]
    ens = Ensemble(members)
    combined = ens.generate_signals(data)["position"]
    expected = 0.5 * (
        members[0].generate_signals(data)["position"]
        + members[1].generate_signals(data)["position"]
    )
    pd.testing.assert_series_equal(combined, expected.clip(-1, 1), check_names=False)


def test_direction_from_position():
    assert Direction.from_position(0.5) is Direction.LONG
    assert Direction.from_position(-0.5) is Direction.SHORT
    assert Direction.from_position(0.0) is Direction.FLAT
