"""Multi-pair universe (a *panel*) for cross-sectional strategies.

A cross-sectional factor ranks pairs against each other at each bar, so it needs
many pairs aligned on one timeline. :class:`Panel` wraps a ``{instrument: bars}``
mapping and exposes **wide** frames (index = time, columns = instrument) for
close / bid / ask / spread and any extra field (e.g. ``rate_diff``).

Pairs may have different histories; the panel aligns on the union of their
timestamps and leaves ``NaN`` where a pair has no bar. Cross-sectional ranking
downstream skips those NaNs, so a pair simply doesn't participate on bars it
lacks.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from forex.data.base import DataProvider, validate_bars


class Panel:
    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        if not frames:
            raise ValueError("Panel needs at least one instrument")
        self._frames: Dict[str, pd.DataFrame] = {
            k: validate_bars(v) for k, v in frames.items()
        }
        self.instruments: List[str] = sorted(self._frames)
        index: Optional[pd.DatetimeIndex] = None
        for frame in self._frames.values():
            index = frame.index if index is None else index.union(frame.index)
        self.index = index.sort_values()

    def _wide(self, column: str) -> pd.DataFrame:
        data = {inst: self._frames[inst].get(column) for inst in self.instruments}
        return pd.DataFrame(data, index=self.index)

    @property
    def close(self) -> pd.DataFrame:
        return self._wide("close")

    @property
    def bid(self) -> pd.DataFrame:
        return self._wide("bid")

    @property
    def ask(self) -> pd.DataFrame:
        return self._wide("ask")

    def spread_frac(self) -> pd.DataFrame:
        """Wide bid/ask spread as a fraction of mid close."""
        close = self.close.replace(0, np.nan)
        return ((self.ask - self.bid) / close).clip(lower=0.0)

    def has_field(self, name: str) -> bool:
        return all(name in self._frames[i].columns for i in self.instruments)

    def field(self, name: str) -> pd.DataFrame:
        if not self.has_field(name):
            missing = [i for i in self.instruments if name not in self._frames[i].columns]
            raise ValueError(f"field {name!r} missing for instruments: {missing}")
        return self._wide(name)

    def upto(self, end: Optional[datetime]) -> "Panel":
        """A new panel truncated to bars at or before ``end`` (warm-up safe)."""
        if end is None:
            return self
        return Panel({k: v[v.index <= end] for k, v in self._frames.items()})

    @classmethod
    def from_provider(
        cls,
        provider: DataProvider,
        instruments: Sequence[str],
        granularity: str = "D",
        *,
        count: int = 500,
        **kwargs,
    ) -> "Panel":
        return cls(
            {
                inst: provider.get_candles(inst, granularity, count=count, **kwargs)
                for inst in instruments
            }
        )
