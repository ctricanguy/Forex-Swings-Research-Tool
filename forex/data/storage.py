"""SQLite persistence.

One single-file database holds three tables:

* ``bars``    -- point-in-time OHLC history (the audit trail of what we knew).
* ``signals`` -- every signal we emitted.
* ``outcomes``-- realised results we attach to signals later.

Point-in-time discipline: a *complete* bar is immutable. We will upsert a bar
that was previously incomplete (the candle finished forming), but we never let
a re-fetch silently rewrite a bar we already recorded as complete. That keeps
backtests honest -- history can't be retroactively "improved".

SQLite is the right default here (zero-config single file, ships with Python,
trivially handles daily/4h volumes on a Raspberry Pi).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from forex.data.base import BAR_COLUMNS, validate_bars
from forex.signals import Direction, Signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    instrument  TEXT    NOT NULL,
    granularity TEXT    NOT NULL,
    time        TEXT    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    bid         REAL,
    ask         REAL,
    volume      REAL,
    complete    INTEGER NOT NULL,
    PRIMARY KEY (instrument, granularity, time)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_time    TEXT    NOT NULL,
    instrument  TEXT    NOT NULL,
    direction   INTEGER NOT NULL,
    strategy    TEXT    NOT NULL,
    entry       REAL,
    stop_loss   REAL,
    take_profit REAL,
    rationale   TEXT,
    created_at  TEXT    NOT NULL,
    UNIQUE (strategy, instrument, bar_time)
);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id   INTEGER NOT NULL REFERENCES signals(id),
    closed_at   TEXT    NOT NULL,
    exit_price  REAL,
    pnl         REAL,
    notes       TEXT,
    PRIMARY KEY (signal_id)
);
"""


class Storage:
    """Thin wrapper around a SQLite database file."""

    def __init__(self, path: Union[str, Path] = "forex.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- bars ---------------------------------------------------------------

    def save_bars(self, instrument: str, granularity: str, df: pd.DataFrame) -> int:
        """Upsert bars. Returns the number of rows written or updated.

        A row already stored as ``complete=1`` is never overwritten; everything
        else (new bars, or previously-incomplete bars now finished) is upserted.
        """
        df = validate_bars(df)
        written = 0
        with closing(self._conn.cursor()) as cur:
            for ts, row in df.iterrows():
                cur.execute(
                    "SELECT complete FROM bars WHERE instrument=? AND granularity=? AND time=?",
                    (instrument, granularity, ts.isoformat()),
                )
                existing = cur.fetchone()
                if existing is not None and existing[0] == 1:
                    continue  # immutable completed bar -- leave it alone
                cur.execute(
                    """
                    INSERT INTO bars (instrument, granularity, time, open, high, low,
                                      close, bid, ask, volume, complete)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(instrument, granularity, time) DO UPDATE SET
                        open=excluded.open, high=excluded.high, low=excluded.low,
                        close=excluded.close, bid=excluded.bid, ask=excluded.ask,
                        volume=excluded.volume, complete=excluded.complete
                    """,
                    (
                        instrument,
                        granularity,
                        ts.isoformat(),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        _opt_float(row["bid"]),
                        _opt_float(row["ask"]),
                        _opt_float(row["volume"]),
                        int(bool(row["complete"])),
                    ),
                )
                written += 1
        self._conn.commit()
        return written

    def load_bars(
        self,
        instrument: str,
        granularity: str,
        *,
        complete_only: bool = True,
    ) -> pd.DataFrame:
        query = (
            "SELECT time, open, high, low, close, bid, ask, volume, complete "
            "FROM bars WHERE instrument=? AND granularity=?"
        )
        if complete_only:
            query += " AND complete=1"
        query += " ORDER BY time ASC"
        df = pd.read_sql_query(query, self._conn, params=(instrument, granularity))
        if df.empty:
            from forex.data.oanda import _empty_frame

            return _empty_frame()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df["complete"] = df["complete"].astype(bool)
        return validate_bars(df[list(BAR_COLUMNS)])

    # -- signals ------------------------------------------------------------

    def save_signal(self, signal: Signal) -> Optional[int]:
        """Persist a signal. Returns its row id, or ``None`` if it already
        existed (same strategy + instrument + bar_time)."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO signals
                    (bar_time, instrument, direction, strategy, entry, stop_loss,
                     take_profit, rationale, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    signal.bar_time.isoformat(),
                    signal.instrument,
                    int(signal.direction),
                    signal.strategy,
                    signal.entry,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.rationale,
                    signal.created_at.isoformat(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid if cur.rowcount else None

    def last_signal(self, strategy: str, instrument: str) -> Optional[Signal]:
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                SELECT bar_time, instrument, direction, strategy, entry, stop_loss,
                       take_profit, rationale, created_at
                FROM signals WHERE strategy=? AND instrument=?
                ORDER BY bar_time DESC LIMIT 1
                """,
                (strategy, instrument),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Signal(
            bar_time=datetime.fromisoformat(row[0]),
            instrument=row[1],
            direction=Direction(row[2]),
            strategy=row[3],
            entry=row[4],
            stop_loss=row[5],
            take_profit=row[6],
            rationale=row[7] or "",
            created_at=datetime.fromisoformat(row[8]),
        )

    def recent_signals(self, limit: int = 20) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY bar_time DESC LIMIT ?",
            self._conn,
            params=(limit,),
        )


def _opt_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)
