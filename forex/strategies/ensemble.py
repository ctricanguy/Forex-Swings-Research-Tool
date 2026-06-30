"""Diversified multi-factor sleeve.

The single most important *positive* finding in the FX factor literature is that
carry, momentum and value are imperfectly (sometimes negatively) correlated, so
a combination is more stable than any one factor alone -- the opposite of the
retail instinct to hunt for one perfect indicator.

:class:`Ensemble` averages the (optionally weighted) positions of its member
strategies. Because each member is point-in-time, their average is too.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from forex.strategies.base import Strategy, positions_frame


class Ensemble(Strategy):
    def __init__(
        self,
        strategies: Sequence[Strategy],
        weights: Optional[Sequence[float]] = None,
        name: str = "ensemble",
    ) -> None:
        if not strategies:
            raise ValueError("Ensemble needs at least one strategy")
        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights length must match strategies length")
        self.strategies = list(strategies)
        self.weights = (
            np.ones(len(strategies))
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        self.weights = self.weights / self.weights.sum()
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        combined = pd.Series(0.0, index=data.index)
        for weight, strat in zip(self.weights, self.strategies):
            combined = combined + weight * strat.generate_signals(data)["position"]
        # Net exposure stays within [-1, 1] since weights sum to 1 and each
        # member position is in [-1, 1].
        return positions_frame(data.index, combined.clip(-1.0, 1.0))
