"""Deterministic synthetic data provider.

Lets the whole pipeline -- strategies, backtester, live ``run_once`` -- run and
be tested with no API keys and no network. It produces a reproducible price
path with a configurable trend and a realistic bid/ask spread, plus an optional
``rate_diff`` column for carry-style strategies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from forex.data.base import DataProvider, validate_bars

# Approximate bar spacing per OANDA-style granularity code.
_GRANULARITY_DELTA = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
}


class SyntheticDataProvider(DataProvider):
    """Generate reproducible bars from a seeded geometric random walk.

    Parameters
    ----------
    start_price:
        Initial mid price.
    drift, vol:
        Per-bar drift and volatility of the random walk (in return space).
    spread_pips, pip:
        Bid/ask spread expressed in pips (``pip`` is the pip size for the pair).
    trend_strength:
        Amplitude of a slow sinusoidal trend overlaid on the walk so that
        trend/momentum strategies have something to find. ``0`` disables it.
    seed:
        RNG seed -- same seed always yields the same bars.
    with_rate_diff:
        If True, attach a slowly varying ``rate_diff`` column (interest-rate
        differential, in percent) for carry strategies.
    """

    def __init__(
        self,
        *,
        start_price: float = 1.1000,
        drift: float = 0.0,
        vol: float = 0.004,
        spread_pips: float = 1.2,
        pip: float = 0.0001,
        trend_strength: float = 0.02,
        seed: int = 7,
        with_rate_diff: bool = False,
    ) -> None:
        self.start_price = start_price
        self.drift = drift
        self.vol = vol
        self.spread_pips = spread_pips
        self.pip = pip
        self.trend_strength = trend_strength
        self.seed = seed
        self.with_rate_diff = with_rate_diff

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
        n = count or 500
        delta = _GRANULARITY_DELTA.get(granularity, timedelta(days=1))
        end_time = end or datetime.now(timezone.utc).replace(microsecond=0)
        times = pd.DatetimeIndex(
            [end_time - delta * (n - 1 - i) for i in range(n)], name="time"
        )

        # Seed deterministically per instrument so different pairs differ but
        # are still reproducible run-to-run.
        rng = np.random.default_rng(self.seed + (hash(instrument) % 10_000))
        shocks = rng.normal(self.drift, self.vol, size=n)
        trend = self.trend_strength * np.sin(np.linspace(0, 3 * np.pi, n))
        log_path = np.cumsum(shocks) + trend
        close = self.start_price * np.exp(log_path)

        # Build OHLC around the close path.
        prev_close = np.concatenate([[self.start_price], close[:-1]])
        bar_range = np.abs(rng.normal(0, self.vol, size=n)) * close
        open_ = prev_close
        high = np.maximum(open_, close) + bar_range / 2
        low = np.minimum(open_, close) - bar_range / 2
        volume = rng.integers(500, 5000, size=n).astype(float)

        half_spread = (self.spread_pips * self.pip) / 2.0
        bid = close - half_spread
        ask = close + half_spread

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "bid": bid,
                "ask": ask,
                "volume": volume,
                "complete": True,
            },
            index=times,
        )

        if self.with_rate_diff:
            # Slow-moving rate differential in percent, sign correlated with the
            # trend so carry and momentum partially agree (as in reality).
            df["rate_diff"] = 1.5 * np.sin(np.linspace(0, 3 * np.pi, n)) + 0.5

        # The most recent bar is the still-forming one unless told otherwise.
        if not complete_only and n > 0:
            df.iloc[-1, df.columns.get_loc("complete")] = False

        return validate_bars(df)
