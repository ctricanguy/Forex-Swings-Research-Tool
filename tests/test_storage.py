from datetime import datetime, timezone

import pandas as pd

from forex.data import SyntheticDataProvider
from forex.data.storage import Storage
from forex.signals import Direction, Signal


def test_bars_roundtrip(tmp_path):
    db = Storage(tmp_path / "t.db")
    df = SyntheticDataProvider().get_candles("EUR_USD", "D", count=50)
    written = db.save_bars("EUR_USD", "D", df)
    assert written == 50
    loaded = db.load_bars("EUR_USD", "D")
    assert len(loaded) == 50
    pd.testing.assert_index_equal(loaded.index, df.index)
    db.close()


def test_complete_bar_is_immutable(tmp_path):
    db = Storage(tmp_path / "t.db")
    df = SyntheticDataProvider().get_candles("EUR_USD", "D", count=10)
    db.save_bars("EUR_USD", "D", df)

    # Attempt to overwrite a completed bar with a tampered close.
    tampered = df.copy()
    tampered.iloc[0, tampered.columns.get_loc("close")] = 9.9999
    db.save_bars("EUR_USD", "D", tampered)

    loaded = db.load_bars("EUR_USD", "D")
    assert loaded["close"].iloc[0] != 9.9999  # original preserved
    db.close()


def test_incomplete_bar_can_be_upgraded(tmp_path):
    db = Storage(tmp_path / "t.db")
    df = SyntheticDataProvider().get_candles("EUR_USD", "D", count=5, complete_only=False)
    db.save_bars("EUR_USD", "D", df)  # last bar incomplete

    # Re-fetch with the bar now complete and a finalised close.
    finalised = df.copy()
    finalised.iloc[-1, finalised.columns.get_loc("complete")] = True
    finalised.iloc[-1, finalised.columns.get_loc("close")] = 1.2345
    db.save_bars("EUR_USD", "D", finalised)

    loaded = db.load_bars("EUR_USD", "D", complete_only=True)
    assert len(loaded) == 5
    assert loaded["close"].iloc[-1] == 1.2345
    db.close()


def test_signal_dedup(tmp_path):
    db = Storage(tmp_path / "t.db")
    sig = Signal(
        bar_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        instrument="EUR_USD",
        direction=Direction.LONG,
        strategy="tsmom_60",
    )
    first = db.save_signal(sig)
    second = db.save_signal(sig)  # same (strategy, instrument, bar_time)
    assert first is not None
    assert second is None  # ignored as duplicate

    last = db.last_signal("tsmom_60", "EUR_USD")
    assert last is not None
    assert last.direction is Direction.LONG
    db.close()
