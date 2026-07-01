"""OANDA v20 data provider (direct REST, no SDK).

Talks to the v20 REST API using only the standard library. The ``oandapyV20``
SDK's setup.py fails to build under modern setuptools (including on Raspberry
Pi OS), and a candle fetch is a plain authenticated GET anyway -- so the SDK
buys nothing but a fragile install. The HTTP layer is injectable, so the whole
provider is tested offline.

Notes from OANDA's docs that shape this implementation:

* ``price=BA`` returns both bid and ask candles, giving us the spread directly
  -- which the backtester needs for honest cost modelling.
* At most 5000 candles per request; we paginate backwards via ``to`` when more
  are requested, and forwards via ``from`` for explicit date ranges.
* Each candle carries ``complete``; the still-forming tail is dropped when
  ``complete_only`` is set, which is the live lookahead guard.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from forex.data.base import DataProvider, validate_bars

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

MAX_CANDLES_PER_REQUEST = 5000

#: Takes a full URL, returns the response body. Injectable for offline tests.
FetchFn = Callable[[str], str]


def _ts(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


class OandaDataProvider(DataProvider):
    """Fetch canonical bars from OANDA v20 over plain REST.

    Parameters
    ----------
    access_token:
        OANDA v20 API token (account page -> "Manage API Access").
    environment:
        ``"practice"`` (demo) or ``"live"``.
    fetch:
        Optional replacement for the HTTP layer (tests inject canned JSON).
    """

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
        self.max_per_request = MAX_CANDLES_PER_REQUEST
        self._fetch = fetch or self._build_http_fetch(access_token)

    def _build_http_fetch(self, token: str) -> FetchFn:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept-Datetime-Format": "RFC3339",
        }

        def fetch(url: str) -> str:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")

        return fetch

    def _request_candles(self, instrument: str, params: dict) -> List[dict]:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}/v3/instruments/{instrument}/candles?{query}"
        body = json.loads(self._fetch(url))
        return [self._parse_candle(c) for c in body.get("candles", [])]

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
        base = {"granularity": granularity, "price": "BA"}
        rows: List[dict] = []

        if start is not None:
            # Forward pagination over [start, end].
            cursor = _ts(start)
            end_ts = _ts(end) if end is not None else None
            while True:
                params = {
                    **base,
                    "from": cursor.isoformat(),
                    "count": self.max_per_request,
                }
                chunk = self._request_candles(instrument, params)
                latest = rows[-1]["time"] if rows else None
                fresh = [c for c in chunk if latest is None or c["time"] > latest]
                rows.extend(fresh)
                if not fresh or len(chunk) < self.max_per_request:
                    break
                cursor = rows[-1]["time"]
                if end_ts is not None and cursor >= end_ts:
                    break
            if end_ts is not None:
                rows = [r for r in rows if r["time"] <= end_ts]
        else:
            # Backward pagination for the most recent `count` candles.
            wanted = count or 500
            to_cursor: Optional[pd.Timestamp] = None
            while len(rows) < wanted:
                params = {**base, "count": min(wanted - len(rows), self.max_per_request)}
                if to_cursor is not None:
                    params["to"] = to_cursor.isoformat()
                chunk = self._request_candles(instrument, params)
                if rows:
                    earliest = min(r["time"] for r in rows)
                    chunk = [c for c in chunk if c["time"] < earliest]
                if not chunk:
                    break
                rows = chunk + rows
                to_cursor = min(r["time"] for r in rows)

        if not rows:
            return validate_bars(_empty_frame())

        df = pd.DataFrame(rows).set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if complete_only:
            df = df[df["complete"]]
        return validate_bars(df)

    @staticmethod
    def _parse_candle(candle: dict) -> dict:
        bid = candle["bid"]
        ask = candle["ask"]
        # Mid OHLC = average of bid and ask OHLC.
        mid = {k: (float(bid[k]) + float(ask[k])) / 2.0 for k in ("o", "h", "l", "c")}
        return {
            "time": _ts(candle["time"]),
            "open": mid["o"],
            "high": mid["h"],
            "low": mid["l"],
            "close": mid["c"],
            "bid": float(bid["c"]),
            "ask": float(ask["c"]),
            "volume": float(candle.get("volume", 0)),
            "complete": bool(candle.get("complete", False)),
        }


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "bid": [],
            "ask": [],
            "volume": [],
            "complete": pd.Series([], dtype=bool),
        },
        index=idx,
    )
