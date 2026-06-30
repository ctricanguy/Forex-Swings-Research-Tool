"""Canonical bar schema and the :class:`DataProvider` interface.

Every provider returns a pandas ``DataFrame`` with this exact shape:

================  ========================================================
index             timezone-aware UTC ``DatetimeIndex`` named ``time``
``open``          mid open price
``high``          mid high price
``low``           mid low price
``close``         mid close price
``bid``           bid price at the bar close (for cost modelling)
``ask``           ask price at the bar close (for cost modelling)
``volume``        tick / traded volume
``complete``      bool -- ``False`` for the still-forming current bar
================  ========================================================

Strategies and the backtester rely on this contract; nothing downstream should
ever reach back into a vendor SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd

#: Required, ordered columns of a canonical bar frame.
BAR_COLUMNS: Sequence[str] = (
    "open",
    "high",
    "low",
    "close",
    "bid",
    "ask",
    "volume",
    "complete",
)


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLC bar. Mostly used at provider boundaries / in tests."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float
    volume: float
    complete: bool


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and lightly normalise) a bar frame against the contract.

    Raises ``ValueError`` on any violation so that contract breakage surfaces
    at the data boundary rather than as a subtle wrong number downstream.
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bar frame index must be a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("bar frame index must be timezone-aware (UTC)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("bar frame index must be sorted ascending by time")
    if df.index.has_duplicates:
        raise ValueError("bar frame index has duplicate timestamps")
    # Reorder to the canonical column order, keeping any extras (e.g. rate_diff)
    extras = [c for c in df.columns if c not in BAR_COLUMNS]
    return df[list(BAR_COLUMNS) + extras]


class DataProvider(ABC):
    """Source of canonical bars.

    The ``complete_only`` flag is the first line of defence against lookahead:
    a swing strategy must act on *closed* candles only, never the bar that is
    still forming. Providers drop the incomplete tail when asked.
    """

    @abstractmethod
    def get_candles(
        self,
        instrument: str,
        granularity: str,
        *,
        count: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        complete_only: bool = True,
    ) -> pd.DataFrame:
        """Return a canonical bar frame for ``instrument`` at ``granularity``.

        Implementations should call :func:`validate_bars` on their result.
        """
        raise NotImplementedError
