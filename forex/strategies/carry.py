"""Carry -- the interest-rate-differential factor.

The oldest and most robust FX anomaly (the forward-premium puzzle): high-rate
currencies tend not to depreciate enough to offset their yield advantage, so
being long the higher-yielder earns a premium. **It is compensation for crash
risk**, not free money -- carry "works until it doesn't", unwinding violently
when global volatility spikes. See ``docs/ARCHITECTURE.md`` Part 2.

This strategy needs an exogenous ``rate_diff`` column (the interest-rate
differential of the pair, in percent) merged onto the bar frame. Go long when
the differential favours the base currency, short when it favours the quote
currency. A ``threshold`` avoids trading on negligible differentials.

Point-in-time: the position at ``t`` uses ``rate_diff[t]`` only. In production,
ensure the rate series is itself lagged to its real publication time before it
reaches here.
"""

from __future__ import annotations

import pandas as pd

from forex.strategies.base import Strategy, positions_frame


class Carry(Strategy):
    def __init__(self, threshold: float = 0.25) -> None:
        self.threshold = threshold
        self.name = "carry"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if "rate_diff" not in data.columns:
            raise ValueError(
                "Carry requires a 'rate_diff' column (interest-rate differential "
                "in percent) on the bar frame. Merge it in from a rates source "
                "(e.g. FRED) before calling generate_signals."
            )
        diff = data["rate_diff"]
        position = pd.Series(0.0, index=data.index)
        position[diff > self.threshold] = 1.0
        position[diff < -self.threshold] = -1.0
        position[diff.isna()] = 0.0
        return positions_frame(data.index, position)
