#!/usr/bin/env python3
"""Incrementally fetch OANDA position-book snapshots into SQLite.

One snapshot per instrument per daily bar close. Resumable: already-stored
snapshots are skipped, so re-running only fills gaps and appends new days.
History depth varies by instrument -- the script walks backwards until the API
stops returning snapshots and reports the actual coverage.

    export OANDA_ACCESS_TOKEN=...
    python scripts/fetch_positioning.py --days 2000        # first run: be patient
    python scripts/fetch_positioning.py --days 30          # daily top-up
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from forex.data.positioning import OandaPositioning
from forex.data.storage import Storage

UNIVERSE = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "USD_CHF", "EUR_GBP",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "EUR_CHF", "EUR_AUD", "GBP_CHF", "NZD_JPY",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=2000)
    parser.add_argument("--db", default="forex.db")
    parser.add_argument("--pause", type=float, default=0.08,
                        help="seconds between requests (rate-limit kindness)")
    parser.add_argument("--miss-limit", type=int, default=30,
                        help="stop walking back after this many consecutive misses")
    args = parser.parse_args()

    token = os.getenv("OANDA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("OANDA_ACCESS_TOKEN is not set")
    client = OandaPositioning(token, os.getenv("OANDA_ENVIRONMENT", "practice"))

    # Daily snapshot times: 21:00 UTC (OANDA's daily candle boundary, on the
    # position book's 20-minute snapshot grid).
    today = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(hours=21)
    times = [today - pd.Timedelta(days=i) for i in range(1, args.days + 1)]

    with Storage(args.db) as storage:
        for instrument in UNIVERSE:
            fetched = skipped = 0
            misses = 0
            earliest = latest = None
            for ts in times:  # newest first, so the miss-limit stops at history end
                if storage.has_positioning(instrument, ts):
                    skipped += 1
                    misses = 0
                    continue
                snap = client.snapshot(instrument, ts)
                _time.sleep(args.pause)
                if snap is None:
                    misses += 1
                    if misses >= args.miss_limit:
                        break
                    continue
                misses = 0
                storage.save_positioning(
                    instrument, snap["time"], snap["long_pct"], snap["short_pct"]
                )
                fetched += 1
                latest = latest or ts
                earliest = ts
            span = f"{earliest:%Y-%m-%d} .. {latest:%Y-%m-%d}" if fetched else "none new"
            print(f"  {instrument}: +{fetched} snapshots ({skipped} already stored) {span}")

    print("done — coverage lives in the 'positioning' table of", args.db)


if __name__ == "__main__":
    main()
