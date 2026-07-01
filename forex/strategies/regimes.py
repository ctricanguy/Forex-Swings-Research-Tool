"""Regime conditioning -- literature-grounded gates, not strategy mining.

The intuition "different markets need different strategies" is documented, but
implemented naively it multiplies the configuration space and dies of
data-snooping. This module keeps it disciplined: two gates, each tied to a
*documented* failure mode, each with almost no free parameters, each returning
NaN scores when inactive so the pair (or the whole book) simply drops out of
the ranking.

* :class:`VolGate` -- global calm/stressed regime from the panel's own
  cross-pair realized volatility. Carry's returns are compensation for crash
  risk: Menkhoff et al. (2012) find global FX vol innovations explain >90% of
  the carry cross-section, and carry crashes when vol spikes. So the grounded
  use is ``VolGate(XSCarry())`` -- carry on when calm, flat when stressed.

* :class:`TrendGate` -- per-pair trending/choppy regime via Kaufman's
  efficiency ratio ER = |P_t - P_{t-w}| / sum(|dP|). Momentum's documented
  failure mode is choppy, mean-reverting tape. ``TrendGate(XSMomentum(...))``
  keeps momentum only in pairs that are actually trending.

Note for FX: "bullish vs bearish" is not a native regime axis -- pairs are
symmetric (short EUR_USD *is* long USD) and momentum already trades both
directions. The FX analogues of that intuition are exactly the two axes above:
risk-on/risk-off (vol) and trending/choppy (ER).

All computations use data at or before each bar; the engine's central one-bar
shift handles execution timing as everywhere else.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from forex.data.panel import Panel
from forex.strategies.cross_sectional import CrossSectionalSignal


def realized_vol(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per-pair trailing return volatility (unannualised; gates compare it only
    against its own history, so scale cancels)."""
    return close.pct_change(fill_method=None).rolling(window).std(ddof=1)


def stress_index(close: pd.DataFrame, window: int = 20) -> pd.Series:
    """Global FX stress: the cross-pair mean of trailing realized vol.

    Needs no external data (no VIX feed) -- the panel itself is the sensor.
    """
    return realized_vol(close, window).mean(axis=1)


def efficiency_ratio(close: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Kaufman's ER in [0, 1]: net move over the window divided by the summed
    path length. ~1/sqrt(window) for a random walk; near 1 for a clean trend."""
    net_move = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net_move / path.replace(0.0, np.nan)


class VolGate(CrossSectionalSignal):
    """Activate a signal only in the calm (or only the stressed) vol regime.

    ``calm`` at bar t means the stress index is at or below its trailing
    ``lookback``-bar ``quantile``. While inactive, every score is NaN -> the
    rank book goes flat (cash), which for carry is exactly the documented
    prescription.
    """

    def __init__(
        self,
        signal: CrossSectionalSignal,
        window: int = 20,
        lookback: int = 252,
        quantile: float = 0.7,
        active_when: str = "calm",
    ) -> None:
        if active_when not in ("calm", "stressed"):
            raise ValueError("active_when must be 'calm' or 'stressed'")
        self.signal = signal
        self.window = window
        self.lookback = lookback
        self.quantile = quantile
        self.active_when = active_when
        self.name = f"volgate[{signal.name}]"

    def score(self, panel: Panel) -> pd.DataFrame:
        scores = self.signal.score(panel)
        stress = stress_index(panel.close, self.window)
        threshold = stress.rolling(
            self.lookback, min_periods=self.lookback // 2
        ).quantile(self.quantile)
        calm = stress <= threshold  # NaN comparisons -> False -> inactive warm-up
        active = calm if self.active_when == "calm" else (~calm & threshold.notna())
        gated = scores.copy()
        gated.loc[~active] = np.nan
        return gated


class EventGate(CrossSectionalSignal):
    """Gate a signal around *scheduled* macro events (FOMC, CPI, NFP, ...).

    The "information circulating" regime: macro announcements account for
    roughly a third of FX price variation, and event windows are when carry
    books get hurt and spreads blow out. Because these events are scheduled
    and published far in advance, gating on the calendar is point-in-time
    legitimate -- at bar t you genuinely know whether tomorrow is an FOMC day.

    ``mode="avoid"``: abstain during event windows (protective -- e.g. take the
    carry book to cash around FOMC). ``mode="only"``: active exclusively inside
    windows (event-conditioned strategies). Windows are calendar days:
    ``[event - before, event + after]``.
    """

    def __init__(
        self,
        signal: CrossSectionalSignal,
        event_dates,
        before: int = 1,
        after: int = 1,
        mode: str = "avoid",
    ) -> None:
        if mode not in ("avoid", "only"):
            raise ValueError("mode must be 'avoid' or 'only'")
        self.signal = signal
        self.event_dates = pd.DatetimeIndex(pd.to_datetime(list(event_dates), utc=True))
        self.before = before
        self.after = after
        self.mode = mode
        self.name = f"eventgate[{signal.name}]"
        blocked = set()
        for event in self.event_dates.normalize():
            for offset in range(-before, after + 1):
                blocked.add(event + pd.Timedelta(days=offset))
        self._blocked = blocked

    def score(self, panel: Panel) -> pd.DataFrame:
        scores = self.signal.score(panel)
        days = panel.index.tz_convert("UTC").normalize()
        in_window = pd.Series([d in self._blocked for d in days], index=panel.index)
        active = ~in_window if self.mode == "avoid" else in_window
        gated = scores.copy()
        gated.loc[~active] = np.nan
        return gated


class TrendGate(CrossSectionalSignal):
    """Activate a signal per pair only where the pair is trending (or choppy).

    ``threshold=None`` defaults to ``2/sqrt(window)`` -- roughly twice the
    random-walk baseline ER, i.e. "meaningfully more directional than noise"
    without introducing a tunable knob.
    """

    def __init__(
        self,
        signal: CrossSectionalSignal,
        window: int = 50,
        threshold: Optional[float] = None,
        active_when: str = "trending",
    ) -> None:
        if active_when not in ("trending", "choppy"):
            raise ValueError("active_when must be 'trending' or 'choppy'")
        self.signal = signal
        self.window = window
        self.threshold = threshold if threshold is not None else 2.0 / math.sqrt(window)
        self.active_when = active_when
        self.name = f"trendgate[{signal.name}]"

    def score(self, panel: Panel) -> pd.DataFrame:
        scores = self.signal.score(panel)
        er = efficiency_ratio(panel.close, self.window)
        trending = er >= self.threshold  # NaN -> False -> inactive warm-up
        active = trending if self.active_when == "trending" else (~trending & er.notna())
        return scores.where(active)
