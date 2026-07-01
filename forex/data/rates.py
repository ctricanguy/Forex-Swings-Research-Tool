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

import json
import os
import time
import urllib.request
from pathlib import Path
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
    timeout: float = 90.0,
    retries: int = 3,
    start: Optional[str] = None,
    api_key: Optional[str] = None,
) -> FetchFn:
    """Default HTTP layer. With an ``api_key`` (free at fred.stlouisfed.org/docs/api)
    it uses the official FRED API -- much faster and more reliable than the
    fredgraph CSV endpoint, which is the fallback. Either way: long timeout,
    retries, optional start date so we don't pull 70 years of history."""

    def fetch(series_id: str) -> str:
        if api_key:
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={api_key}&file_type=json"
            )
            if start:
                url += f"&observation_start={start}"
        else:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            if start:
                url += f"&cosd={start}"
        request = urllib.request.Request(url, headers={"User-Agent": "forex-tool/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    return with_retries(fetch, retries=retries)


def parse_fred_api_json(text: str) -> pd.Series:
    """Parse the official FRED API observations JSON to a Series."""
    body = json.loads(text)
    dates, values = [], []
    for obs in body.get("observations", []):
        raw = obs.get("value", ".").strip()
        if raw in (".", ""):
            continue
        dates.append(pd.Timestamp(obs["date"], tz="UTC"))
        values.append(float(raw))
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="time")).sort_index()


def _parse_any(text: str) -> pd.Series:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return parse_fred_api_json(text)
    return parse_fredgraph_csv(text)


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
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = "data/rates_cache",
        cache_max_age_days: float = 3.0,
    ) -> None:
        self.series = {**DEFAULT_SERIES, **(series or {})}
        api_key = api_key or os.getenv("FRED_API_KEY")
        self._fetch = fetch or _http_fetch_factory(timeout, retries, start, api_key)
        self.lag_bars = lag_bars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_max_age_days = cache_max_age_days
        self._cache: Dict[str, pd.Series] = {}

    def _disk_cache_path(self, currency: str) -> Optional[Path]:
        return self.cache_dir / f"{currency}.csv" if self.cache_dir else None

    def _load_disk_cache(self, currency: str) -> Optional[pd.Series]:
        """Fresh cached rates, or any manually dropped-in CSV as last resort.

        Cache files are ``date,value`` CSVs -- you can also create them by hand
        (download the series from FRED in a browser) if the network is hostile.
        """
        path = self._disk_cache_path(currency)
        if path is None or not path.exists():
            return None
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days > self.cache_max_age_days:
            return None
        df = pd.read_csv(path)
        idx = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0], utc=True), name="time")
        return pd.Series(df.iloc[:, 1].astype(float).values, index=idx).sort_index()

    def _save_disk_cache(self, currency: str, series: pd.Series) -> None:
        path = self._disk_cache_path(currency)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": series.index.date, "value": series.values}).to_csv(
            path, index=False
        )

    def rate(self, currency: str) -> pd.Series:
        currency = currency.upper()
        if currency not in self._cache:
            if currency not in self.series:
                raise KeyError(
                    f"no FRED series configured for {currency}; "
                    f"known: {sorted(self.series)}"
                )
            cached = self._load_disk_cache(currency)
            if cached is not None:
                self._cache[currency] = cached
            else:
                fetched = _parse_any(self._fetch(self.series[currency]))
                self._save_disk_cache(currency, fetched)
                self._cache[currency] = fetched
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
