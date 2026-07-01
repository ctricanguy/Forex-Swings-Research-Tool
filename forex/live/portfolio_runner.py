"""Live cross-sectional portfolio runner.

The service-facing sibling of :class:`~forex.live.runner.LiveRunner`. On each
cycle it fetches completed bars for the whole universe, optionally attaches real
rate differentials, scores the cross-section with the **same**
``CrossSectionalSignal`` code the backtester validates, builds the long/short
book with the **same** ``rank_weights``, optionally applies the vol-target scale
from the risk layer, and -- only when the book actually changed -- persists it
and pushes a formatted rebalance alert.

Parity with :class:`~forex.backtest.portfolio.PortfolioBacktest`: the weights
computed on the latest completed bar ``t`` are the position held from now on,
exactly mirroring the engine's ``weights.shift(1)`` convention.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

from forex.backtest.engine import CostModel
from forex.backtest.metrics import periods_per_year
from forex.backtest.portfolio import PortfolioBacktest, rank_weights
from forex.data.base import DataProvider
from forex.data.panel import Panel
from forex.data.storage import Storage
from forex.live.notifier import ConsoleNotifier, Notifier
from forex.strategies.cross_sectional import CrossSectionalSignal

DISCLAIMER = "Research signal — not investment advice."


def format_book_message(
    name: str,
    bar_time,
    weights: Dict[str, float],
    previous: Optional[Dict[str, float]] = None,
) -> str:
    lines = [f"[{name}] portfolio rebalance | bar={bar_time:%Y-%m-%d %H:%M UTC}"]
    for instrument, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        if abs(weight) < 1e-9:
            continue
        lines.append(f"  {instrument:<9} {weight:+.1%}")
    if previous:
        turnover = sum(
            abs(weights.get(k, 0.0) - previous.get(k, 0.0))
            for k in set(weights) | set(previous)
        )
        lines.append(f"  turnover vs previous book: {turnover:.1%}")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


class PortfolioLiveRunner:
    def __init__(
        self,
        provider: DataProvider,
        signal: CrossSectionalSignal,
        storage: Storage,
        *,
        instruments: Sequence[str],
        granularity: str = "D",
        history_bars: int = 400,
        quantile: float = 1.0 / 3.0,
        gross_leverage: float = 1.0,
        notifier: Optional[Notifier] = None,
        rebalance_tolerance: float = 1e-6,
        rates=None,  # optional forex.data.rates.FredRates
        costs: Optional[CostModel] = None,
        vol_target: Optional[float] = None,
        vol_window: int = 60,
        max_leverage: float = 3.0,
    ) -> None:
        self.provider = provider
        self.signal = signal
        self.storage = storage
        self.instruments = list(instruments)
        self.granularity = granularity
        self.history_bars = history_bars
        self.quantile = quantile
        self.gross_leverage = gross_leverage
        self.notifier = notifier or ConsoleNotifier()
        self.rebalance_tolerance = rebalance_tolerance
        self.rates = rates
        self.costs = costs or CostModel()
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.max_leverage = max_leverage

    def _build_panel(self) -> Optional[Panel]:
        frames = {}
        for instrument in self.instruments:
            df = self.provider.get_candles(
                instrument,
                self.granularity,
                count=self.history_bars,
                complete_only=True,
            )
            if df.empty:
                continue
            self.storage.save_bars(instrument, self.granularity, df)
            frames[instrument] = df
        if not frames:
            return None
        if self.rates is not None:
            frames = self.rates.attach(frames)
        return Panel(frames)

    def _vol_scale(self, panel: Panel) -> float:
        """Trailing-vol scale for the next holding, same math as RiskOverlay."""
        if not self.vol_target:
            return 1.0
        returns = PortfolioBacktest(
            self.signal,
            panel,
            costs=self.costs,
            granularity=self.granularity,
            quantile=self.quantile,
            gross_leverage=self.gross_leverage,
        ).run().returns
        window = returns.iloc[-self.vol_window:]
        if len(window) < 2:
            return 1.0
        realized = window.std(ddof=1) * math.sqrt(periods_per_year(self.granularity))
        if realized <= 0:
            return 1.0
        return float(min(self.vol_target / realized, self.max_leverage))

    def run_once(self) -> Optional[Dict[str, float]]:
        """One cycle. Returns the new book if a rebalance was emitted, else None."""
        panel = self._build_panel()
        if panel is None:
            return None

        scores = self.signal.score(panel)
        weights_frame = rank_weights(scores, self.quantile, self.gross_leverage)
        latest = weights_frame.iloc[-1].fillna(0.0) * self._vol_scale(panel)
        bar_time = panel.index[-1].to_pydatetime()
        book = {inst: float(w) for inst, w in latest.items()}

        previous = self.storage.last_book(self.signal.name)
        if previous is not None:
            _, prev_weights = previous
            drift = max(
                abs(book.get(k, 0.0) - prev_weights.get(k, 0.0))
                for k in set(book) | set(prev_weights)
            )
            if drift <= self.rebalance_tolerance:
                return None  # book unchanged -> stay quiet

        self.storage.save_book(self.signal.name, bar_time, book)
        prev_weights = previous[1] if previous else None
        self.notifier.send(
            format_book_message(self.signal.name, bar_time, book, prev_weights)
        )
        return book

    def start(self, cron: str = "5 0 * * *") -> None:  # pragma: no cover
        """Block and run ``run_once`` on a cron schedule (requires APScheduler)."""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as exc:
            raise ImportError(
                "APScheduler is required for PortfolioLiveRunner.start(). "
                "Install it with `pip install APScheduler`."
            ) from exc
        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(self.run_once, CronTrigger.from_crontab(cron))
        self.notifier.send(f"Portfolio live runner started ({self.signal.name}).")
        scheduler.start()
