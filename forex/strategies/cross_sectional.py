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
from typing import Optional, Sequence

import numpy as np
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


class XSValue(CrossSectionalSignal):
    """Cross-sectional value: long-horizon reversal (cheap = fell the most).

    Asness, Moskowitz & Pedersen (2013) measure currency value as the ~5-year
    reversal in the *real* exchange rate. With price-only data, the standard
    practitioner proxy is the negative of the long-horizon nominal return:
    pairs that fell the most over ``lookback`` bars score highest (cheapest).

    If ``field`` is given (e.g. a real-exchange-rate gap where positive means
    undervalued), that panel field is used directly instead of the price proxy.
    Value is deliberately slow-moving and low-turnover -- cost-friendly, and
    negatively correlated with momentum, which is why the combination works.
    """

    def __init__(self, lookback: int = 1000, field: Optional[str] = None) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.field = field
        self.name = f"xsvalue_{field or lookback}"

    def score(self, panel: Panel) -> pd.DataFrame:
        if self.field is not None:
            return panel.field(self.field)
        close = panel.close
        return -(close / close.shift(self.lookback) - 1.0)


class XSEnsemble(CrossSectionalSignal):
    """Diversified multi-factor sleeve, cross-sectional edition.

    Each member's scores are z-scored *across pairs per bar* (so a rate
    differential in percent and a trailing return in decimals become
    comparable), then combined as a weighted average.

    NaN handling is the ``require_all`` switch:

    * ``require_all=True`` (default): NaNs propagate -- a pair only gets an
      ensemble score on bars where every member can score it. Honest during
      warm-up, strict.
    * ``require_all=False``: a NaN member simply *abstains* and the remaining
      members are re-weighted (weighted mean over available scores). This is
      the right semantics for regime-gated members (a gated-off factor should
      hand its vote to the others, not silence the pair). When every member is
      present the two modes agree exactly.
    """

    def __init__(
        self,
        signals: Sequence[CrossSectionalSignal],
        weights: Optional[Sequence[float]] = None,
        name: str = "xs_ensemble",
        require_all: bool = True,
    ) -> None:
        if not signals:
            raise ValueError("XSEnsemble needs at least one signal")
        if weights is not None and len(weights) != len(signals):
            raise ValueError("weights length must match signals length")
        self.signals = list(signals)
        w = np.ones(len(signals)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()
        self.name = name
        self.require_all = require_all

    def score(self, panel: Panel) -> pd.DataFrame:
        total: Optional[pd.DataFrame] = None
        available: Optional[pd.DataFrame] = None
        for weight, signal in zip(self.weights, self.signals):
            raw = signal.score(panel)
            mu = raw.mean(axis=1)
            sd = raw.std(axis=1, ddof=1).replace(0.0, np.nan)
            z = raw.sub(mu, axis=0).div(sd, axis=0)
            if self.require_all:
                total = weight * z if total is None else total + weight * z
            else:
                contrib = weight * z.fillna(0.0)
                mass = weight * z.notna().astype(float)
                total = contrib if total is None else total + contrib
                available = mass if available is None else available + mass
        if self.require_all:
            return total
        return total.div(available.replace(0.0, np.nan))
