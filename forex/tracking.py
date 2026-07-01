"""Signal outcome tracking -- the verifiable track record.

If this is ever sold as a service, the single most valuable asset is an
*auditable* log: every signal timestamped when emitted (the ``signals`` table),
marked to market after the fact, never retroactively edited. This module closes
that loop.

Convention: consecutive signals for the same ``(strategy, instrument)`` define
position spans. A span opens at signal ``i`` (entry = its close price) and
closes at signal ``i+1`` (exit = that signal's entry price). Spans with
direction 0 (flat) close the previous span but open nothing. The most recent
signal stays open and is reported separately.

PnL is computed on mid prices; real fills pay spread on both sides, so treat
these numbers as an upper bound (the honest-costs backtest remains the place
where costs are modelled).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from forex.data.storage import Storage


def update_outcomes(storage: Storage) -> int:
    """Mark closed spans to market. Idempotent. Returns spans recorded."""
    signals = storage.all_signals()
    if signals.empty:
        return 0
    recorded = 0
    for (_, _), group in signals.groupby(["strategy", "instrument"], sort=False):
        group = group.sort_values("bar_time")
        rows = group.to_dict("records")
        for current, nxt in zip(rows, rows[1:]):
            if current["direction"] == 0 or current["entry"] is None:
                continue
            if nxt["entry"] is None:
                continue
            pnl = (
                current["direction"]
                * (nxt["entry"] - current["entry"])
                / current["entry"]
            )
            storage.record_outcome(
                current["id"], nxt["bar_time"], nxt["entry"], float(pnl)
            )
            recorded += 1
    return recorded


def open_positions(storage: Storage) -> pd.DataFrame:
    """The latest signal per (strategy, instrument) with a nonzero direction."""
    signals = storage.all_signals()
    if signals.empty:
        return signals
    latest = signals.groupby(["strategy", "instrument"], sort=False).tail(1)
    return latest[latest["direction"] != 0].reset_index(drop=True)


def track_record(storage: Storage) -> str:
    """Human-readable performance digest per strategy."""
    outcomes = storage.outcomes()
    lines = ["Track record (closed signal spans, mid-price marks)"]
    if outcomes.empty:
        lines.append("  no closed spans yet")
    else:
        for strategy, group in outcomes.groupby("strategy"):
            pnl = group["pnl"].astype(float)
            wins = (pnl > 0).sum()
            lines.append(
                f"  {strategy}: {len(pnl)} closed | win rate {wins / len(pnl):.0%} | "
                f"avg {pnl.mean():+.2%} | cum {pnl.sum():+.2%}"
            )
    open_df = open_positions(storage)
    if not open_df.empty:
        lines.append("Open positions:")
        for _, row in open_df.iterrows():
            side = "LONG" if row["direction"] > 0 else "SHORT"
            entry = f" @ {row['entry']:.5f}" if pd.notna(row["entry"]) else ""
            lines.append(
                f"  {row['strategy']} {row['instrument']} {side}{entry} "
                f"since {row['bar_time'][:10]}"
            )
    return "\n".join(lines)
