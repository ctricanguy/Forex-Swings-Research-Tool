"""Live signal layer.

Thin, auditable runners that, on a schedule, fetch the latest *completed* bars,
call the **same** ``forex.strategies`` code the backtester validates, and push
signals out. No auto-execution -- this is an alert/research tool by design.

* :class:`LiveRunner` -- per-pair strategies, one signal per direction change.
* :class:`PortfolioLiveRunner` -- a cross-sectional long/short book, one alert
  per rebalance.
"""

from forex.live.notifier import ConsoleNotifier, Notifier, TelegramNotifier
from forex.live.portfolio_runner import PortfolioLiveRunner, format_book_message
from forex.live.runner import LiveRunner

__all__ = [
    "ConsoleNotifier",
    "Notifier",
    "TelegramNotifier",
    "LiveRunner",
    "PortfolioLiveRunner",
    "format_book_message",
]
