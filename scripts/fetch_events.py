#!/usr/bin/env python3
"""Fetch official macro release calendars from FRED into data/events/*.csv.

Covers CPI and the Employment Situation (NFP). FOMC decision dates are not a
FRED release -- add them by hand as data/events/fomc.csv (one date per line,
header 'date') from federalreserve.gov/monetarypolicy/fomccalendars.htm; the
schedule is published years ahead.

    export FRED_API_KEY=...        # free key: fred.stlouisfed.org/docs/api/api_key.html
    python scripts/fetch_events.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.data.events import FRED_RELEASES, fetch_release_dates, save_event_dates


def main() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY is not set (free at fred.stlouisfed.org/docs/api/api_key.html)"
        )
    out_dir = Path("data/events")
    for name, release_id in FRED_RELEASES.items():
        dates = fetch_release_dates(release_id, api_key, start="2015-01-01")
        recent = dates[dates >= "2015-01-01"]
        save_event_dates(recent, out_dir / f"{name}.csv")
        print(f"  {name}: {len(recent)} release dates -> {out_dir / f'{name}.csv'}")
    print("done — EventGate loads these via forex.data.events.load_event_dates('data/events')")


if __name__ == "__main__":
    main()
