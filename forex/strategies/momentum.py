"""Time-series (trend) momentum.

The CTA staple and one of the four documented cross-asset FX premia (carry,
momentum, value, trend). Position is the sign of the trailing return over a
lookback window: if the pair has risen over the last ``lookback`` bars, go long;
if it has fallen, go short. A ``deadband`` keeps us flat when the move is
negligible, cutting churn (and therefore spread cost).

This is point-in-time by construction: ``close[t] / close[t-lookback] - 1`` uses
only information available at bar ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex.strategies.base import Strategy, positions_frame


class TimeSeriesMomentum(Strategy):
    def __init__(self, lookback: int = 60, deadband: float = 0.0) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.deadband = deadband
        self.name = f"tsmom_{lookback}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        trailing_return = close / close.shift(self.lookback) - 1.0
        position = pd.Series(0.0, index=data.index)
        position[trailing_return > self.deadband] = 1.0
        position[trailing_return < -self.deadband] = -1.0
        # No trailing return available until we have `lookback` bars of history.
        position[trailing_return.isna()] = 0.0
        return positions_frame(data.index, position)
