#!/usr/bin/env python3
"""End-to-end smoke test of the OANDA v20 provider.

Requires OANDA_ACCESS_TOKEN (and optionally OANDA_ENVIRONMENT=practice|live) in
the environment or a .env-sourced shell. Exits nonzero on any failure so it can
gate deployments.

    export OANDA_ACCESS_TOKEN=...   # practice-account token
    python scripts/smoke_oanda.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.data.base import validate_bars


def main() -> int:
    token = os.getenv("OANDA_ACCESS_TOKEN")
    if not token:
        print("FAIL: OANDA_ACCESS_TOKEN is not set")
        return 2
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    from forex.data.oanda import OandaDataProvider

    provider = OandaDataProvider(token, environment)
    for instrument in ("EUR_USD", "USD_JPY"):
        df = provider.get_candles(instrument, "D", count=30)
        validate_bars(df)
        if df.empty:
            print(f"FAIL: no candles for {instrument}")
            return 1
        if not df["complete"].all():
            print(f"FAIL: incomplete bars leaked for {instrument}")
            return 1
        spread = ((df["ask"] - df["bid"]) / df["close"]).mean()
        print(
            f"OK {instrument}: {len(df)} daily bars through "
            f"{df.index[-1]:%Y-%m-%d}, close={df['close'].iloc[-1]:.5f}, "
            f"avg spread={spread:.5%}"
        )
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
