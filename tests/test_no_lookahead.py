"""Lookahead-bias tests -- the rigor linchpin of the whole project."""

import pandas as pd

from forex.backtest import detect_lookahead
from forex.data import SyntheticDataProvider
from forex.strategies import MovingAverageCross, TimeSeriesMomentum
from forex.strategies.base import Strategy, positions_frame


def _data(n=400):
    return SyntheticDataProvider().get_candles("EUR_USD", "D", count=n)


class _Cheater(Strategy):
    """A strategy that peeks at the *next* bar's return -- pure lookahead.

    Point-in-time-violating on purpose, to prove the detector catches it.
    """

    name = "cheater"

    def generate_signals(self, data):
        future_return = data["close"].shift(-1) / data["close"] - 1.0
        position = future_return.apply(lambda x: 1.0 if x > 0 else -1.0)
        return positions_frame(data.index, position.fillna(0.0))


def test_honest_strategies_have_no_leak():
    data = _data()
    for strat in (TimeSeriesMomentum(lookback=40), MovingAverageCross(10, 30)):
        diag = detect_lookahead(strat, data)
        assert diag["leaking_bars"] == 0.0
        assert diag["max_position_diff"] == 0.0


def test_detector_catches_future_peeking():
    data = _data()
    diag = detect_lookahead(_Cheater(), data, probe=25)
    # Every probed bar's position changes once future candles are withheld.
    assert diag["leaking_bars"] > 0
    assert diag["max_position_diff"] > 0.0
