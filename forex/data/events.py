"""Scheduled-event calendars (CPI, NFP, FOMC, ...).

Calendars live as plain CSVs under ``data/events/`` -- one file per event type,
first column a date. Two ways to populate them:

* ``scripts/fetch_events.py`` pulls official release dates from the FRED API
  (CPI = release 10, Employment Situation/NFP = release 50). Needs the free
  ``FRED_API_KEY``.
* Drop in a hand-made CSV (e.g. FOMC decision dates from
  federalreserve.gov/monetarypolicy/fomccalendars.htm) -- central bank
  calendars are published years ahead, so this is point-in-time legitimate.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

#: FRED release ids for the calendars we care about.
FRED_RELEASES = {
    "cpi": 10,            # Consumer Price Index
    "nfp": 50,            # Employment Situation (Non-Farm Payrolls)
}


def load_event_dates(path: Union[str, Path]) -> pd.DatetimeIndex:
    """Load event dates from one CSV file or every CSV in a directory."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    dates: list = []
    for file in files:
        df = pd.read_csv(file)
        dates.extend(pd.to_datetime(df.iloc[:, 0], utc=True).tolist())
    return pd.DatetimeIndex(sorted(set(dates)))


def fetch_release_dates(
    release_id: int,
    api_key: str,
    *,
    start: Optional[str] = None,
    timeout: float = 60.0,
) -> pd.DatetimeIndex:
    """Official release dates from the FRED API."""
    url = (
        "https://api.stlouisfed.org/fred/release/dates"
        f"?release_id={release_id}&api_key={api_key}&file_type=json"
        "&include_release_dates_with_no_data=true&sort_order=asc&limit=1000"
    )
    if start:
        url += f"&realtime_start={start}"
    request = urllib.request.Request(url, headers={"User-Agent": "forex-tool/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dates = [pd.Timestamp(d["date"], tz="UTC") for d in body.get("release_dates", [])]
    return pd.DatetimeIndex(sorted(set(dates)))


def save_event_dates(dates: Iterable, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": [pd.Timestamp(d).date() for d in dates]})
    frame.to_csv(path, index=False)
