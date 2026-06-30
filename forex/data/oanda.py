"""OANDA v20 data provider.

Wraps the ``oandapyV20`` client. Imports of the SDK are deferred to
construction so the rest of the package (and the test suite) works without the
dependency installed; you only need it when you actually fetch live data.

Notes from OANDA's docs that shape this implementation:

* ``price="BA"`` returns both bid and ask candles, giving us the spread
  directly -- which the backtester needs for honest cost modelling.
* ``InstrumentsCandlesFactory`` auto-paginates around the 5000-candle limit.
* The candle endpoint exposes ``complete`` per candle; we drop the incomplete
  tail when ``complete_only`` is set, which is the live lookahead guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from forex.data.base import DataProvider, validate_bars


class OandaDataProvider(DataProvider):
    """Fetch canonical bars from OANDA v20.

    Parameters
    ----------
    access_token:
        OANDA v20 API token (from the account's "Manage API Access" page).
    environment:
        ``"practice"`` (demo) or ``"live"``.
    """

    def __init__(self, access_token: str, environment: str = "practice") -> None:
        try:
            from oandapyV20 import API
        except ImportError as exc:  # pragma: no cover - exercised only on real use
            raise ImportError(
                "oandapyV20 is required for OandaDataProvider. "
                "Install it with `pip install oandapyV20`."
            ) from exc
        self._api = API(access_token=access_token, environment=environment)

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
        from oandapyV20.contrib.factories import InstrumentsCandlesFactory

        params: dict = {"granularity": granularity, "price": "BA"}
        if start is not None:
            params["from"] = start.isoformat()
        if end is not None:
            params["to"] = end.isoformat()
        if count is not None and start is None:
            # OANDA rejects count alongside an explicit from/to range.
            params["count"] = count

        rows: list[dict] = []
        for request in InstrumentsCandlesFactory(instrument=instrument, params=params):
            self._api.request(request)
            for candle in request.response.get("candles", []):
                rows.append(self._parse_candle(candle))

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
            "time": pd.Timestamp(candle["time"]).tz_convert("UTC"),
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
