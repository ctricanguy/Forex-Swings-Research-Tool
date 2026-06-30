"""Cross-sectional factor signals.

Where a per-pair :class:`~forex.strategies.base.Strategy` returns a position for
one instrument's time series, a :class:`CrossSectionalSignal` returns a **score
per pair per bar** -- a number that is only meaningful *relative* to the other
pairs at the same time. The portfolio engine then ranks those scores to build a
long/short book (long the most attractive, short the least).

This matches how carry and momentum are actually documented in the literature:
as cross-sectional premia (long high-yield / high-momentum currencies, short the
others), not single-pair indicator rules.

Scores must be point-in-time: the score at bar ``t`` may use only data at
``<= t``. The one-bar execution shift is applied later, centrally, by the
portfolio engine -- never here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from forex.data.panel import Panel


class CrossSectionalSignal(ABC):
    """Maps a :class:`~forex.data.panel.Panel` to a wide score frame.

    The returned frame is indexed by time with one column per instrument; a
    higher score means "more attractive to be long". ``NaN`` means the pair has
    no opinion / no data on that bar and is excluded from ranking.
    """

    name: str = "xs_base"

    @abstractmethod
    def score(self, panel: Panel) -> pd.DataFrame:
        raise NotImplementedError


class XSMomentum(CrossSectionalSignal):
    """Cross-sectional momentum: rank by trailing return over ``lookback`` bars."""

    def __init__(self, lookback: int = 60) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.name = f"xsmom_{lookback}"

    def score(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        return close / close.shift(self.lookback) - 1.0


class XSCarry(CrossSectionalSignal):
    """Cross-sectional carry: rank by the interest-rate differential per pair.

    Requires a ``rate_diff`` field on every instrument in the panel.
    """

    name = "xs_carry"

    def score(self, panel: Panel) -> pd.DataFrame:
        if not panel.has_field("rate_diff"):
            raise ValueError(
                "XSCarry requires a 'rate_diff' field on every instrument in the "
                "panel (interest-rate differential, in percent)."
            )
        return panel.field("rate_diff")
