"""Real interest-rate differentials from FRED (no API key required).

Uses the public ``fredgraph.csv`` endpoint
(``https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES``), which needs no
registration. The fetcher is injectable so tests run fully offline and you can
swap in the official FRED API (with a key) later without touching callers.

The default series map uses short/policy rates per currency. **Verify each
series before trusting it with money**: FRED ids get discontinued (several OECD
interbank series stopped updating around 2023-24). Override any entry via the
``series`` argument.

Point-in-time discipline: rates are forward-filled onto the bar timeline and
then lagged ``lag_bars`` (default 1), so the differential attached to bar ``t``
was publicly known before ``t``.
"""

from __future__ import annotations

import time
import urllib.request
from typing import Callable, Dict, Optional

import pandas as pd

#: FRED series per currency. Daily where available, monthly OECD otherwise.
#: VERIFY before live use -- see module docstring.
DEFAULT_SERIES: Dict[str, str] = {
    "USD": "DFF",                # Effective Federal Funds Rate (daily)
    "EUR": "ECBDFR",             # ECB Deposit Facility Rate (daily)
    "GBP": "IUDSOIA",            # SONIA overnight rate (daily)
    "JPY": "IRSTCI01JPM156N",    # Japan immediate call rate (monthly, OECD)
    "CHF": "IR3TIB01CHM156N",    # CHF 3m interbank (monthly, OECD)
    "CAD": "IRSTCB01CAM156N",    # Canada central bank rate (monthly, OECD)
    "AUD": "IR3TIB01AUM156N",    # AUD 3m interbank (monthly, OECD)
    "NZD": "IR3TIB01NZM156N",    # NZD 3m interbank (monthly, OECD)
}

FetchFn = Callable[[str], str]


def with_retries(fn: Callable, retries: int = 3, base_delay: float = 2.0) -> Callable:
    """Wrap a callable with exponential-backoff retries (FRED can be slow)."""

    def wrapped(*args):
        for attempt in range(retries):
            try:
                return fn(*args)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))

    return wrapped


def _http_fetch_factory(
    timeout: float = 90.0, retries: int = 3, start: Optional[str] = None
) -> FetchFn:
    """Default HTTP layer: long timeout, retries, and an optional ``cosd`` start
    date so we don't pull 70 years of history when bars begin in 2016."""

    def fetch(series_id: str) -> str:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        if start:
            url += f"&cosd={start}"
        request = urllib.request.Request(url, headers={"User-Agent": "forex-tool/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    return with_retries(fetch, retries=retries)


def parse_fredgraph_csv(text: str) -> pd.Series:
    """Parse fredgraph CSV (``DATE,SERIES`` header; ``.`` = missing) to a Series."""
    dates, values = [], []
    for line in text.strip().splitlines()[1:]:  # skip header
        parts = line.split(",")
        if len(parts) < 2:
            continue
        raw = parts[1].strip()
        if raw in (".", ""):
            continue
        dates.append(pd.Timestamp(parts[0].strip(), tz="UTC"))
        values.append(float(raw))
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="time")).sort_index()


class FredRates:
    """Fetch, cache, and attach per-pair rate differentials.

    ``rate_diff`` for ``EUR_USD`` = EUR short rate - USD short rate, in percent
    per annum -- exactly the field :class:`~forex.strategies.cross_sectional.XSCarry`
    and the portfolio engine's financing model expect.
    """

    def __init__(
        self,
        series: Optional[Dict[str, str]] = None,
        fetch: Optional[FetchFn] = None,
        lag_bars: int = 1,
        *,
        start: Optional[str] = None,
        timeout: float = 90.0,
        retries: int = 3,
    ) -> None:
        self.series = {**DEFAULT_SERIES, **(series or {})}
        self._fetch = fetch or _http_fetch_factory(timeout, retries, start)
        self.lag_bars = lag_bars
        self._cache: Dict[str, pd.Series] = {}

    def rate(self, currency: str) -> pd.Series:
        currency = currency.upper()
        if currency not in self._cache:
            if currency not in self.series:
                raise KeyError(
                    f"no FRED series configured for {currency}; "
                    f"known: {sorted(self.series)}"
                )
            self._cache[currency] = parse_fredgraph_csv(self._fetch(self.series[currency]))
        return self._cache[currency]

    def rate_diff(self, instrument: str, index: pd.DatetimeIndex) -> pd.Series:
        """Differential (base - quote) aligned to ``index``, ffilled and lagged."""
        base_ccy, quote_ccy = instrument.split("_")
        base, quote = self.rate(base_ccy), self.rate(quote_ccy)
        union = base.index.union(quote.index)
        diff = base.reindex(union).ffill() - quote.reindex(union).ffill()
        aligned = diff.sort_index().reindex(index, method="ffill")
        return aligned.shift(self.lag_bars)

    def attach(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Return a copy of ``{instrument: bars}`` with ``rate_diff`` added."""
        out: Dict[str, pd.DataFrame] = {}
        for instrument, df in frames.items():
            df = df.copy()
            df["rate_diff"] = self.rate_diff(instrument, df.index)
            out[instrument] = df
        return out
