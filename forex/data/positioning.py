"""OANDA retail positioning (position book) -- the order-flow-adjacent data a
retail account can actually access.

Institutional order flow (Evans & Lyons) is the documented flow edge, but its
data (EBS/Refinitiv) is not retail-accessible. What *is* accessible: OANDA
publishes aggregate client positioning per instrument, with **historical
snapshots** via the ``time`` parameter on
``/v3/instruments/{instrument}/positionBook``. The practitioner literature uses
it contrarian: when the retail crowd is heavily long, fade it.

Honest caveats, before anyone trades this:
* Sample = OANDA's retail clients only, not the market.
* Documented predictive power for position-based sentiment is weak-to-moderate;
  academic studies of positioning indices often find no significant returns.
  Treat this as an *orthogonal information source to test*, not a known edge.
* Snapshot history depth varies by instrument; the fetcher records whatever
  exists and the validation slices to actual coverage.

Snapshots are stored point-in-time in SQLite (``positioning`` table) by
``scripts/fetch_positioning.py`` and attached to bar frames as a
``net_long_pct`` field. A snapshot at bar close time is information available
at that close; the engine's central one-bar shift handles execution as always.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable, Dict, Optional

import pandas as pd

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

FetchFn = Callable[[str], str]


class OandaPositioning:
    """Fetch position-book snapshots (aggregate % of clients long/short)."""

    def __init__(
        self,
        access_token: str,
        environment: str = "practice",
        fetch: Optional[FetchFn] = None,
        timeout: float = 30.0,
    ) -> None:
        if environment not in _HOSTS:
            raise ValueError(f"environment must be one of {sorted(_HOSTS)}")
        self.base_url = _HOSTS[environment]
        self.timeout = timeout
        self._fetch = fetch or self._build_http_fetch(access_token)

    def _build_http_fetch(self, token: str) -> FetchFn:
        headers = {"Authorization": f"Bearer {token}"}

        def fetch(url: str) -> str:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")

        return fetch

    def snapshot(
        self, instrument: str, time: Optional[datetime] = None
    ) -> Optional[Dict]:
        """One snapshot: ``{"time": Timestamp, "long_pct": float, "short_pct": float}``.

        Returns None when no snapshot exists for that time (history boundary).
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/positionBook"
        if time is not None:
            url += "?" + urllib.parse.urlencode({"time": pd.Timestamp(time).isoformat()})
        try:
            body = json.loads(self._fetch(url))
        except Exception:
            return None
        book = body.get("positionBook")
        if not book:
            return None
        long_pct = sum(float(b["longCountPercent"]) for b in book.get("buckets", []))
        short_pct = sum(float(b["shortCountPercent"]) for b in book.get("buckets", []))
        if long_pct + short_pct <= 0:
            return None
        return {
            "time": pd.Timestamp(book["time"]).tz_convert("UTC"),
            "long_pct": long_pct,
            "short_pct": short_pct,
        }


def attach_positioning(
    storage, frames: Dict[str, pd.DataFrame], *, max_gap_bars: int = 5
) -> Dict[str, pd.DataFrame]:
    """Attach stored snapshots to bar frames as ``net_long_pct`` (0-100).

    Forward-fills across small gaps (weekends/missing snapshots) up to
    ``max_gap_bars``; beyond that the field is NaN and the pair simply drops
    out of positioning-based rankings for those bars.
    """
    out: Dict[str, pd.DataFrame] = {}
    for instrument, df in frames.items():
        series = storage.load_positioning(instrument)
        df = df.copy()
        if series.empty:
            df["net_long_pct"] = float("nan")
        else:
            union = series.index.union(df.index)
            aligned = series.reindex(union).ffill(limit=max_gap_bars)
            df["net_long_pct"] = aligned.reindex(df.index)
        out[instrument] = df
    return out
