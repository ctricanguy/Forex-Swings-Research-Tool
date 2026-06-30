"""Moving-average crossover.

The textbook trend rule, included as a second concrete, price-only strategy.
Long when the fast SMA is above the slow SMA, short when below. Treat it as a
teaching example, not an edge: pure technical rules like this mostly fail once
you correct for data-snooping (see ``docs/ARCHITECTURE.md`` Part 2). It exists
here to exercise the framework and to be a baseline a real factor must beat.

Point-in-time: both moving averages at bar ``t`` are computed from closes at
``<= t`` only.
"""

from __future__ import annotations

import pandas as pd

from forex.strategies.base import Strategy, positions_frame


class MovingAverageCross(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"ma_cross_{fast}_{slow}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        position = pd.Series(0.0, index=data.index)
        position[fast_ma > slow_ma] = 1.0
        position[fast_ma < slow_ma] = -1.0
        # Flat until both MAs are defined.
        position[slow_ma.isna()] = 0.0
        return positions_frame(data.index, position)
