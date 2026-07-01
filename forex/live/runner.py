"""Live runner.

On each invocation (``run_once``) it, per instrument:

1. fetches the latest *completed* bars from the data provider,
2. persists them point-in-time to SQLite,
3. calls the shared strategy to get the desired position at the latest bar,
4. emits a :class:`~forex.signals.Signal` *only if* the direction changed since
   the last persisted signal, then stores it.

Parity with the backtester: the backtester holds ``desired.shift(1)`` -- i.e.
it acts on the bar *after* the decision bar. Live mirrors this exactly: when
bar ``t`` closes we read ``desired[t]`` and that becomes the position held from
now on. Same code, same convention, no divergence.

``start()`` wraps ``run_once`` in an APScheduler cron job; that import is
deferred so the module (and tests) load without APScheduler installed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from forex.data.base import DataProvider
from forex.data.storage import Storage
from forex.live.notifier import ConsoleNotifier, Notifier
from forex.signals import Direction, Signal
from forex.strategies.base import Strategy


class LiveRunner:
    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        storage: Storage,
        *,
        instruments: Sequence[str],
        granularity: str = "D",
        history_bars: int = 400,
        notifier: Optional[Notifier] = None,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.storage = storage
        self.instruments = list(instruments)
        self.granularity = granularity
        self.history_bars = history_bars
        self.notifier = notifier or ConsoleNotifier()

    def run_once(self) -> List[Signal]:
        """One polling cycle. Returns the signals newly emitted this cycle."""
        emitted: List[Signal] = []
        for instrument in self.instruments:
            signal = self._process_instrument(instrument)
            if signal is not None:
                emitted.append(signal)
        return emitted

    def _process_instrument(self, instrument: str) -> Optional[Signal]:
        df = self.provider.get_candles(
            instrument,
            self.granularity,
            count=self.history_bars,
            complete_only=True,  # never decide on the still-forming bar
        )
        if df.empty:
            return None
        self.storage.save_bars(instrument, self.granularity, df)

        desired = self.strategy.generate_signals(df)["position"]
        latest_position = float(desired.iloc[-1])
        bar_time = df.index[-1].to_pydatetime()
        direction = Direction.from_position(latest_position)

        previous = self.storage.last_signal(self.strategy.name, instrument)
        if previous is not None and previous.direction == direction:
            return None  # no change -> stay quiet

        signal = Signal(
            bar_time=bar_time,
            instrument=instrument,
            direction=direction,
            strategy=self.strategy.name,
            entry=float(df["close"].iloc[-1]),
            rationale=f"desired position {latest_position:+.2f} on {self.granularity} close",
        )
        if self.storage.save_signal(signal) is not None:
            self.notifier.send_signal(signal)
        return signal

    def start(self, cron: str = "5 0 * * *") -> None:  # pragma: no cover
        """Block and run ``run_once`` on a cron schedule (requires APScheduler)."""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as exc:
            raise ImportError(
                "APScheduler is required for LiveRunner.start(). "
                "Install it with `pip install APScheduler`."
            ) from exc

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(self.run_once, CronTrigger.from_crontab(cron))
        self.notifier.send("Forex live runner started.")
        scheduler.start()
