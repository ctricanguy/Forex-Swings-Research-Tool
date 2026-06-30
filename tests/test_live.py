from datetime import datetime

import pandas as pd

from forex.data import SyntheticDataProvider
from forex.data.base import DataProvider, validate_bars
from forex.data.storage import Storage
from forex.live import ConsoleNotifier, LiveRunner
from forex.signals import Direction
from forex.strategies import TimeSeriesMomentum


class FixedProvider(DataProvider):
    """Returns a preset frame, so we control exactly what 'latest bar' is."""

    def __init__(self, df: pd.DataFrame):
        self.df = validate_bars(df)

    def get_candles(self, instrument, granularity, *, count=None, start=None,
                    end=None, complete_only=True):
        df = self.df
        if complete_only and "complete" in df.columns:
            df = df[df["complete"].astype(bool)]
        return df


def _runner(provider, storage):
    return LiveRunner(
        provider,
        TimeSeriesMomentum(lookback=20),
        storage,
        instruments=["EUR_USD"],
        granularity="D",
        notifier=ConsoleNotifier(),
    )


def test_run_once_emits_latest_desired_position(tmp_path):
    data = SyntheticDataProvider().get_candles("EUR_USD", "D", count=200)
    storage = Storage(tmp_path / "t.db")
    runner = _runner(FixedProvider(data), storage)

    emitted = runner.run_once()
    assert len(emitted) == 1

    strat = TimeSeriesMomentum(lookback=20)
    desired = strat.generate_signals(data)["position"]
    expected = Direction.from_position(desired.iloc[-1])
    assert emitted[0].direction is expected
    # bar_time is the latest completed bar.
    assert emitted[0].bar_time == data.index[-1].to_pydatetime()
    storage.close()


def test_run_once_quiet_when_direction_unchanged(tmp_path):
    data = SyntheticDataProvider().get_candles("EUR_USD", "D", count=200)
    storage = Storage(tmp_path / "t.db")
    runner = _runner(FixedProvider(data), storage)

    first = runner.run_once()
    assert len(first) == 1
    # Same data again -> direction unchanged -> nothing emitted.
    second = runner.run_once()
    assert second == []
    # Only the first cycle pushed a message.
    assert len(runner.notifier.messages) == 1
    storage.close()


def test_run_once_persists_bars(tmp_path):
    data = SyntheticDataProvider().get_candles("EUR_USD", "D", count=120)
    storage = Storage(tmp_path / "t.db")
    runner = _runner(FixedProvider(data), storage)
    runner.run_once()
    stored = storage.load_bars("EUR_USD", "D")
    assert len(stored) == 120
    storage.close()
